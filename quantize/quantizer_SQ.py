import torch
import inspect
import logging
import functools
import torch.nn as nn
from torch.quantization import quantize
from torchsummary import summary
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from awq.utils.calib_data import CalibrationCallback, get_calib_dataset, custom_multimodal_dataset, get_calib_dataset_dm, Mean_Max_Activation_Hook, apply_hook, run_calibration, remove_hooks

from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import clear_memory, get_best_device
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)
from awq.quantize.fake_quant import WxAxLinear, WxAxConv2d
import matplotlib.pyplot as plt
import os

class SqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        quantise_act,
        weight_quant_conv_type,
        weight_quant_type,
        act_quant_conv_type,
        act_quant_conv_group_size,
        w_bit,
        wv_bit,
        a_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        quant_act=False,
        apply_clip=True,
        applyScale=True,
        samples=512,
        processor=None,
        calib_data_type="",
        blocksize=512,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
        LLM_ViT_serial=True,
        quantVision=False,
        quantText=True,
        quantVisionProjection=False,
        quantTextProjection=False,
        quantUnet = True,
        quantTextEncoder = False,
        quantVAE = False,
        quantTransformer = False,
        diffusion_model = False,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.wv_bit = wv_bit
        self.a_bit = a_bit
        self.quantise_act = quantise_act
        self.weight_quant_conv_type = weight_quant_conv_type
        self.weight_quant_type = weight_quant_type
        self.act_quant_conv_group_size = act_quant_conv_group_size
        self.act_quant_conv_type = act_quant_conv_type
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.quant_act = quant_act
        self.apply_clip = apply_clip
        self.applyScale = applyScale
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.samples=samples
        self.processor=processor
        self.calib_data_type=calib_data_type
        self.blocksize=blocksize
        self.LLM_ViT_serial=LLM_ViT_serial        
        self.quantVision=quantVision
        self.quantText=quantText
        self.quantVisionProjection=quantVisionProjection
        self.quantTextProjection=quantTextProjection
        self.quantUnet = quantUnet
        self.quantTextEncoder = quantTextEncoder
        self.quantVAE = quantVAE
        self.quantTransformer = quantTransformer
        self.diffusion_model = diffusion_model
        # self.modules, self.module_kwargs, self.inps = self.init_quant(
        #     n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        # )
        self.modules, self.module_kwargs, self.inps = self.init_quant(n_samples=self.samples,seqlen=self.blocksize,
            LLM_ViT_serial = self.LLM_ViT_serial , quantVision = self.quantVision , quantText = self.quantText,
            quantVisionProjection = self.quantVisionProjection, quantTextProjection = self.quantTextProjection )
        self.moduleNames = self.awq_model.get_debugModuleNames(quantVision = self.quantVision , quantText = self.quantText,
                                quantVisionProjection = self.quantVisionProjection, quantTextProjection = self.quantTextProjection )
        self.scalingStates = self.awq_model.get_scalingStates(quantVision = self.quantVision , quantText = self.quantText,
                                quantVisionProjection = self.quantVisionProjection, quantTextProjection = self.quantTextProjection )
        self.projectionNames = self.awq_model.get_projectionNames(quantVision = self.quantVision , quantText = self.quantText,
                                quantVisionProjection = self.quantVisionProjection, quantTextProjection = self.quantTextProjection )
    
    class MyTraversal():
        def __init__(self):
            self.name = None
            self.parent = None
            self.lin_conv = []
        
        def traverse(self, name, module, parent):
            self.name = name
            self.parent = parent
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                self.lin_conv.append((parent, name, module))

            for name, child in module.named_children():
                if child is not None:
                    self.traverse(name, child, module)
        
        def get_lin_conv(self):
            return self.lin_conv

    def pseudo_quantize_tensor(self, w: torch.Tensor, bitWidth):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**bitWidth - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (bitWidth - 1) - 1
            min_int = -(2 ** (bitWidth - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self, quantPath, debugPlot=False):
        if self.diffusion_model == False:
            if debugPlot == True:
                os.makedirs(quantPath, exist_ok=True)
                moduleNames = self.moduleNames

            for j in range(len(self.modules)):

                # translate ViT outputs and text embeddings into input for LLM in the case when architecture is serial like in LlaVa
                self.inps = None

                for i in tqdm(range(len(self.modules[j])), desc="SQ "  + self.moduleNames[j] ):
                    # Move module and inputs to correct device
                    common_device = next(self.modules[j][i].parameters()).device
                    if common_device is None or str(common_device) == "cpu":
                        if torch.cuda.is_available():
                            best_device = "cuda:" + str(i % torch.cuda.device_count())
                        else:
                            best_device = get_best_device()

                        self.modules[j][i] = self.modules[j][i].to(best_device)
                        common_device = next(self.modules[j][i].parameters()).device

                    if self.module_kwargs[j].get("position_ids") is not None:
                        self.module_kwargs[j]["position_ids"] = self.module_kwargs[j]["position_ids"].to(common_device)

                    if self.module_kwargs[j].get("attention_mask") is not None:
                        self.module_kwargs[j]["attention_mask"] = self.module_kwargs[j]["attention_mask"].to(common_device)

                    if self.inps is not None:
                        self.inps = self.inps.to(common_device)

                    # We need to move the rotary embedding every time we move to a new module.
                    # Transformers 4.45.0 moved rotary embedding to model definition as of this PR:
                    # https://github.com/huggingface/transformers/pull/32617
                    self.awq_model.move_embed(self.model, common_device)

                    for k, v in self.module_kwargs[j].items():
                        # position embeddings found in tuple
                        if isinstance(v, tuple):
                            self.module_kwargs[j][k] = tuple(
                                item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) 
                                else item for item in v
                            )

                    # [STEP 1]: Get layer, extract linear modules, extract input features
                    named_linears = get_named_linears(self.modules[j][i])

                    if len(named_linears) == 1 and list(named_linears.keys()) == ['']:
                        named_linears[ self.projectionNames[j] ] = self.modules[j][i]
                        named_linears.pop('')
                    elif len(named_linears) > 1 and list(named_linears.keys()) == ['']:
                        raise Exception("Make sure the multiple linear layers have corresponding names for next steps!")

                    # Filter out the linear layers we don't want to exclude
                    named_linears = exclude_layers_to_not_quantize(named_linears, self.modules_to_not_convert)

                    input_feat , act_scales = self._get_input_feat(self.modules[j][i], named_linears , j)
                    clear_memory()

                    # [STEP 2]: Compute SQ scale list
                    if debugPlot == True:
                        figVec = []
                        axesVec = []
                        for name, linear_layer in named_linears.items():
                            fig, axes = plt.subplots(3, 1)
                            figVec.append( fig )
                            axesVec.append( axes )
                            axes[0].hist(torch.flatten( linear_layer.weight.data.cpu() ), bins=100, color='blue', alpha=0.7, edgecolor='black')
                            axes[0].set_title( "Before scaling "+moduleNames[j]+str(i)+" "+name )

                    # [STEP 3]: Apply SQ scale list
                    if self.scalingStates[j] == True and self.applyScale == True:
                        if 'vision' in self.moduleNames[j].lower():
                            module_config: List[Dict] = self.awq_model.get_layers_for_scaling_vision_SQ(
                                self.modules[j][i], act_scales
                            )
                        elif 'text' in self.moduleNames[j].lower():
                            module_config: List[Dict] = self.awq_model.get_layers_for_scaling_SQ(
                                self.modules[j][i], act_scales
                            )
                        for layer in module_config:
                            self.smooth_ln_fcs(ln=layer['prev_op'], fcs=layer['layers'], act_scales=layer['input_scales'],  alpha=0.5)

                    if debugPlot == True:
                        count = 0
                        for name, linear_layer in named_linears.items():
                            axesVec[count][1].hist(torch.flatten( linear_layer.weight.data.cpu() ), bins=100, color='red', alpha=0.7, edgecolor='black')
                            axesVec[count][1].set_ylabel( "After scaling" )
                            count = count + 1

                    # [STEP 4]: Quantize weights
                    if 'vision' in self.moduleNames[j].lower():
                        bitWidth = self.wv_bit
                    elif 'text' in self.moduleNames[j].lower():
                        bitWidth = self.w_bit
                    else:
                        raise Exception("Should not have come here!!")

                    if not self.export_compatible:
                        self._apply_quant(self.modules[j][i], named_linears, bitWidth)
                    else:
                        if not self.quant_act:
                            self._apply_quant_fake(self.modules[j][i], named_linears, bitWidth)
                        else:
                            self._apply_quant_fake_act(self.modules[j][i], named_linears, bitWidth)

                    if debugPlot == True:
                        count = 0
                        for name, linear_layer in named_linears.items():
                            axesVec[count][2].hist(torch.flatten( linear_layer.weight.data.cpu() ), bins=100, color='green', alpha=0.7, edgecolor='black')
                            axesVec[count][2].set_xlabel( "After quantization "+moduleNames[j]+str(i)+" "+name )
                            save_path = os.path.join(quantPath, "SQ layer "+moduleNames[j]+str(i)+" "+name.replace("." , " ") )
                            figVec[count].savefig(save_path , dpi=300)
                            plt.close( figVec[count] )
                            count = count + 1

                    clear_memory()
        else:
            print("Inside SQuantizer, printing module list, passed by Object Class")
            calib_num_infer_steps = 50
            smoothing_blocks = self.awq_model.get_smoothing_blocks()
            hook_d = self.apply_hooks_to_smoothing_blocks(smoothing_blocks)

            samples = get_calib_dataset_dm(model_pipeline = self.awq_model.get_pipeline(),
                                        text_dataset = "clip-benchmark/wds_mscoco_captions2017",
                                        batch_size = 8,
                                        n_samples = 96,
                                        seed = 42,
                                        device = "cuda",
                                        split = "test",
                                        text_column = "txt")

            callback = None
            run_calibration(self.awq_model.get_pipeline(), samples, callback, calib_num_infer_steps)
            
            for block_name, block_module in smoothing_blocks.items():
                print(f"Applying Smoothing to {block_name}")
                scaling_groups = self.awq_model.get_layers_for_scaling_unet(block_module, hook_d[block_name])
                for group in scaling_groups:
                    prev_op = group['prev_op']
                    layers_to_scale = group['layers']
                    mean_of_max_act = group['activations_max']

                    self.smooth_ln_fcs(prev_op, layers_to_scale, mean_of_max_act[0],alpha=0.80)
                
                print(f"Clearing and removing hook for {block_name}")
                for layers, hooks in hook_d[block_name].items():
                    hooks.clear()
                    hooks.hook_handle.remove()
                del hook_d[block_name]
                torch.cuda.empty_cache()

            for component_type, component_lists in self.modules.items():
                if not component_lists:  # Skip empty components, "vae" : []
                    continue
        
                for component_idx, module_list in enumerate(component_lists):
                    # Set component identifier in case there are multiple components
                    # unet --> unet_1, unet_2, unet_3, text_encoder --> text_encoder_1, text_encoder_2...
                    component_name = f"{component_type}_{component_idx + 1}" if len(component_lists) > 1 else component_type
                    root = self.awq_model.get_root(component_type, component_idx)
                    print(f"Quantising {component_name}")
                    self.awq_model.set_quantized_components(component_name)
                    
                    for i, (name, mod) in enumerate(tqdm(module_list, desc=f"Processing {component_name}")):
                        try:
                            common_device = next(mod.parameters()).device
                        except StopIteration:
                            continue

                        if common_device is None or str(common_device) == "cpu":
                            best_device = f"cuda:{i % torch.cuda.device_count()}" if torch.cuda.is_available() else get_best_device()
                            mod = mod.to(best_device)

                        # Find and quantize layers
                        module_traversal = self.MyTraversal()
                        module_traversal.traverse(name, mod, root)
                        lin_and_conv_layers = module_traversal.get_lin_conv()
                        # lin_and_conv_layers = exclude_layers_to_not_quantize(lin_and_conv_layers, self.modules_to_not_convert)
                        
                        if not lin_and_conv_quantizelayers:
                            continue
                        
                        #SmoothQuant Logic:
                        
                        self._apply_quant_fake_act(mod, lin_and_conv_layers, self.w_bit)

        

    @torch.no_grad()
    def smooth_ln_fcs(self, ln, fcs, act_scales, model_type = "transformers", alpha=0.5):
        if not isinstance(fcs, list):
            fcs = [fcs]
        for fc in fcs:
            #assert isinstance(fc, nn.Linear), f"layer is {fc}"
            # print("Prev_op: ", ln)
            # print("Prev_op numel: ", ln.weight.numel())
            # print("Layer: ", fc.in_features)
            # print("Activations: ", act_scales.numel())
            # print(" ")
            assert ln.weight.numel() == fc.in_features == act_scales.numel()
        # if model_type == "llama":
        #     assert isinstance(ln, LlamaRMSNorm)
        # elif model_type == "baichuan":
        #     assert isinstance(ln, RMSNorm)
        # elif model_type == "mixtral":
        #     assert isinstance(ln, MixtralRMSNorm)
        # else:
        #     assert isinstance(ln, nn.LayerNorm)

        device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)
        weight_scales = torch.cat([fc.weight.abs().max(
            dim=0, keepdim=True)[0] for fc in fcs], dim=0)
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

        scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
                ).clamp(min=1e-5).to(device).to(dtype)

        ln.weight.div_(scales)
        # if model_type == "transformers":
        if hasattr(ln , "bias"):
            ln.bias.div_(scales)

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))

    # def pack(self):
    #     for i in tqdm(range(len(self.modules)), desc="Packing"):
    #         named_linears = get_named_linears(self.modules[i])
    #         named_linears = exclude_layers_to_not_quantize(
    #             named_linears, self.modules_to_not_convert
    #         )
    #         self._apply_quant(self.modules[i], named_linears)
    #         clear_memory()

    def _apply_quant_fake(self, module, named_linears: Dict[str, nn.Linear], bitWidth):
        if self.diffusion_model == False:
            for name, linear_layer in named_linears.items():
                # NOTE: small regression in perplexity if linear layer uses .cpu().float()
                linear_layer = linear_layer.to(get_best_device()).half()

                linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                    linear_layer.weight.data, bitWidth
                )
                linear_layer.cpu()
                clear_memory()
        else:
            raise Exception("Not Implemented for Diffusion Models yet")

    def _apply_quant_fake_act(self, module, named_linears: Dict[str, nn.Linear], bitWidth):
        if self.diffusion_model == False:
            for name, linear_layer in named_linears.items():
                # NOTE: small regression in perplexity if linear layer uses .cpu().float()
                linear_layer = linear_layer.to(get_best_device()).half()

                # modify this based on names of transformer layers of underlying network
                quantize_bmm_input = 'k_proj' in name or 'v_proj' in name or 'q_proj' in name

                fakeLayer = WxAxLinear.from_float(
                    linear_layer,
                    weight_quant='group',
                    act_quant='per_token',
                    quantize_output=quantize_bmm_input,
                    n_bits_W=bitWidth,
                    n_bits_A=self.a_bit,
                    group_size_W=self.group_size,
                )

                linear_layer.cpu()
                fakeLayer.to(next(module.parameters()).device)
                if not isinstance( module , nn.Linear ):
                    set_op_by_name(module, name, fakeLayer)
                else:
                    set_op_by_name(self.awq_model.model, name, fakeLayer)
                clear_memory()
        else:
            for parent, name, layer in named_linears:

                quantize_bmm_input = 'k_proj' in name or 'v_proj' in name or 'q_proj' in name
                
                if isinstance(layer, torch.nn.Linear):
                    fakeLayer = WxAxLinear.from_float(
                        layer,
                        weight_quant= self.weight_quant_type,
                        act_quant='per_token',
                        quantize_output = quantize_bmm_input,
                        n_bits_W=bitWidth,
                        n_bits_A=self.a_bit,
                        group_size_W=self.group_size,
                        # codeBookQuantInd=self.codeBookQuantInd,
                    )
                    layer.cpu()
                    fakeLayer.to(next(module.parameters()).device)
                    setattr(parent, name, fakeLayer)
                    
                elif isinstance(layer, torch.nn.Conv2d):

                    fakeLayer = WxAxConv2d.from_float(
                        layer,
                        weight_quant= self.weight_quant_conv_type,
                        act_quant = self.act_quant_conv_type,
                        quantize_output = self.quantise_act,
                        act_group_size = self.act_quant_conv_group_size,
                        n_bits_W = bitWidth,
                        n_bits_A = self.a_bit,
                        # codeBookQuantInd=self.codeBookQuantInd,
                    )
                    layer.cpu()
                    fakeLayer.to(next(module.parameters()).device)
                    setattr(parent, name, fakeLayer)


    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear], bitWidth):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data
            )

            if self.version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            elif self.version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast

            else:
                raise ValueError(f"Unknown version {self.version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=bitWidth,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            if not isinstance( module , nn.Linear ):
                set_op_by_name(module, name, q_linear)
            else:
                set_op_by_name(self.awq_model.model, name, q_linear)
            clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    # def init_quant(self, n_samples=128, max_seq_len=512):
    def init_quant(self, n_samples=128, seqlen=512, LLM_ViT_serial=True, quantVision=True, quantText=True, quantVisionProjection=False, quantTextProjection=False):
    # LLM_ViT_serial True means image passed through ViT and then is merged with text tokens before going to LLM

        # check to make sure only those components are quantized which exist for this particular model
        if self.diffusion_model == False:
            self.awq_model.checkQuantStatus( quantVision, quantText, quantVisionProjection, quantTextProjection)

            # make list of modules to be used during quant process
            modules = []
            if quantVision is True:
                modules.append( self.awq_model.get_model_layers_vision(self.model) )
            if quantVisionProjection is True:
                modules.append( self.awq_model.get_model_layers_visionProjection(self.model) )
            if quantText is True:
                modules.append( self.awq_model.get_model_layers(self.model) )
            if quantTextProjection is True:
                modules.append( self.awq_model.get_model_layers_textProjection(self.model) )

            if self.calib_data_type=="multimodal":

                # get calibration data
                data_dict=custom_multimodal_dataset(data_dict=self.calib_data)
                totensor = self.awq_model.get_toTensor()

                for i in totensor:
                    if data_dict[i] is not None:
                        data_dict[i]= data_dict[i].to(next(self.model.parameters()).device)
                        if data_dict[i].shape[0] != n_samples:
                            self.samples = data_dict[i].shape[0]
                            print( 'Setting number of samples equal to size of calibration data size....' )
                inps = []
                layer_kwargs = []
        
                best_device = get_best_device()
                # modules[0] = modules[0].to(best_device)
                self.awq_model.move_embed(self.model, best_device)
        
                # get input and kwargs to layer 0
                # with_kwargs is only supported in PyTorch 2.0
                # use this Catcher hack for now
                # class Catcher(nn.Module):
                #     def __init__(self, module):
                #         super().__init__()
                #         self.module = module
        
                #     def forward(self, *args, **kwargs):
                #         # assume first input to forward is hidden states
                #         if len(args) > 0:
                #             hidden_states = args[0]
                #             del args
                #         else:
                #             first_key = list(kwargs.keys())[0]
                #             hidden_states = kwargs.pop(first_key)
        
                #         inps.append(hidden_states)
                #         layer_kwargs.update(kwargs)
                #         raise ValueError  # early exit to break later inference
        
                # patch layer 0 to catch input and kwargs
                # modules[0] = Catcher(modules[0])
                # modules[0] = modules[0].module  # restore
                
                # for key in layer_kwargs.keys():
                #     if key in data_dict.keys():
                #         del layer_kwargs[key]
                # print('l1',layer_kwargs)

                # translate inputs from data_dict to layer_kwargs
                if quantVision is True:
                    # written with the CLIP vision encoder in mind -- will need to be changed if some VLM has some other vision encoder
                    layer_kwargs.append( self.model.prepare_inputs_for_generation(input_ids=data_dict['pixel_values'], inputs_embeds = data_dict['image_embeds'], cache_position = data_dict['cache_position'], attention_mask=None, causal_attention_mask=None) )
                    if 'attention_mask' not in layer_kwargs[0].keys():
                        layer_kwargs[0]["attention_mask"] = None
                if quantVisionProjection is True:
                    layer_kwargs.append( dict( inputs_embeds = data_dict['image_embeds_projection'] ) )
                if quantText is True:
                    if 'input_ids' in data_dict.keys():
                        layer_kwargs.append( self.model.prepare_inputs_for_generation(**data_dict) )
                    else:
                        layer_kwargs.append( self.model.prepare_inputs_for_generation(None,**data_dict) )
                if quantTextProjection is True:
                    layer_kwargs.append( dict( inputs_embeds = data_dict['text_embeds_projection'] ) )

                # layer_kwargs = self.model.prepare_inputs_for_generation(**layer_kwargs, input_ids=data_dict['pixel_values'], inputs_embeds = data_dict['image_embeds'], attention_mask=None, causal_attention_mask=None)
                # if 'attention_mask' not in layer_kwargs.keys():
                #     layer_kwargs["attention_mask"] = None

                # if 'input_ids' in data_dict.keys():
                #     layer_kwargs = self.model.prepare_inputs_for_generation(**layer_kwargs,**data_dict)
                # else:
                #     layer_kwargs = self.model.prepare_inputs_for_generation(None, **layer_kwargs,**data_dict)

                # print('l2',layer_kwargs)
                inps = None

                # modules[0] = modules[0].cpu()
                self.awq_model.move_embed(self.model, "cpu")
                clear_memory()
                for layer_kwarg in layer_kwargs:
                    if layer_kwarg.get("attention_mask") is not None:
                        layer_kwarg["attention_mask"] = layer_kwarg["attention_mask"].to(
                            best_device
                        )
                # if layer_kwargs.get("attention_mask") is not None:
                #     layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                #         best_device
                #     )
            else:
                samples = get_calib_dataset(
                    data=self.calib_data,
                    tokenizer=self.tokenizer,
                    n_samples=n_samples,
                    max_seq_len=seqlen,
                    split=self.split,
                    text_column=self.text_column,
                )
                samples = torch.cat(samples, dim=0)

                inps = []
                layer_kwargs = {}

                best_device = get_best_device()
                modules[0] = modules[0].to(best_device)
                self.awq_model.move_embed(self.model, best_device)

                # get input and kwargs to layer 0
                # with_kwargs is only supported in PyTorch 2.0
                # use this Catcher hack for now
                class Catcher(nn.Module):
                    def __init__(self, module):
                        super().__init__()
                        self.module = module

                    def forward(self, *args, **kwargs):
                        # assume first input to forward is hidden states
                        if len(args) > 0:
                            hidden_states = args[0]
                            del args
                        else:
                            first_key = list(kwargs.keys())[0]
                            hidden_states = kwargs.pop(first_key)

                        inps.append(hidden_states)
                        layer_kwargs.update(kwargs)
                        raise ValueError  # early exit to break later inference

                # patch layer 0 to catch input and kwargs
                modules[0] = Catcher(modules[0])
                try:
                    self.model(samples.to(next(self.model.parameters()).device))
                except ValueError:  # work with early exit
                    pass
                modules[0] = modules[0].module  # restore

                # Update the layer kwargs with `prepare_inputs_for_generation` method
                # that takes care of everything to avoid unexpected errors.
                layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
                # Pop the input_ids as they are not needed at all.
                layer_kwargs.pop("input_ids")

                del samples
                inps = inps[0]

                modules[0] = modules[0].cpu()
                self.awq_model.move_embed(self.model, "cpu")

                clear_memory()

                if layer_kwargs.get("attention_mask") is not None:
                    layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                        best_device
                    )

            return modules, layer_kwargs, inps

        else:
            modules = {"unet": [], "text_encoder":[], "vae":[], "transformer":[]}
            hooked_modules = {} #contains module_name:{layer_name: hook, ...., layer_N_name: hook}

            if self.quantUnet is True:
                modules["unet"] = self.awq_model.get_model_layers
                hooked_modules["unet"] = apply_hook(self.awq_model.get_unet())
              
            if self.quantTextEncoder is True:
                print("Trying to get TE")
                modules["text_encoder"] = self.awq_model.get_model_layers_te()
            if self.quantVAE is True:
                modules["vae"] = self.awq_model.get_model_layers_vae()
            if self.quantTransformer is True:
                modules["transformer"] = self.awq_model.get_model_layers_transformers()
                hooked_modules["transformer"] = apply_hook(self.awq_model.get_transformer())
            
            best_device = get_best_device()
            #SEED = 42
            
            # Calibration Dataset:
            #samples = get_calib_dataset_dm(
            #        model_pipeline=self.awq_model.get_pipeline(),
            #        text_dataset = "clip-benchmark/wds_mscoco_captions2017",
            #        n_samples=n_samples,
            #        generator = torch.Generator(best_device).manual_seed(SEED),
            #        split="test",
            #        text_column="txt",
            #        batch_size = 4,
            #) 

            #callback_fn = CalibrationCallback() #For TimeStep data ---> [0: 981, 1: 962, .... ]
            #run_calibration(self.awq_model.get_pipeline, samples, callback_fn)
            layer_kwargs = []
            inps = []

        return modules, layer_kwargs, inps
    

    def apply_hooks_to_smoothing_blocks(self, blocks):
        hook_d = {}
        for name, block in blocks.items():
            print(f"Applying Hooks to {name}")
            hook_d[name] = apply_hook(block)
        
        return hook_d


    def _get_input_feat(self, layer, named_linears , j):
        # get activation scales for SQ
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        hooks = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
            hooks.append(
                named_linears[name].register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

        if self.inps is not None:
            self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        else:
            self.inps=self.module_kwargs[j]['inputs_embeds'].to(torch.float16).to(next(layer.parameters()).device)
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs[j], layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        for h in hooks:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat , act_scales

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
