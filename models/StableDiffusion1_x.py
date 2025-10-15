from .base import torch
from .base import BaseAWQForDiffusion
from .base import QUANTISABLE_COMPONENTS
from diffusers.models.attention import BasicTransformerBlock

class StableDiffusion1_x(BaseAWQForDiffusion):

    def __init__(self, pipeline, model_type, is_quantized, config, quant_config, access_token, refiner_path = None):
        if refiner_path is not None:
            raise Exception("StableDiffusion1.5 has no refiner model, if there is its not supported")
    
        super().__init__(pipeline, model_type, is_quantized, config, quant_config)
        print("A StableDiffusion1_x model has been created!")
        self.quantizable_components = {"unet": [], "text_encoder":[], "vae":[], "transformer":[]}
        self.quantized_components = []
        self.set_quantizable_components()


    def set_quantizable_components(self):
        all_components = self.pipeline.components.keys()
        for component in all_components:

            if "unet" in component:
                self.quantizable_components["unet"].append(component)
            
            elif "text_encoder" in component:
                self.quantizable_components["text_encoder"].append(component)
            
            elif "vae" in component:
                self.quantizable_components["vae"].append(component)
            
            elif "transformer" in component:
                self.quantizable_components["transformer"].append(component)
            
    def checkQuantStatus(self, quantUnet = True, quantTextEncoder = False, quantVAE = False, quantTransformer = True):
        if quantTransformer == True:
            raise Exception("There is no Transformer in this Diffusion Model")
    
    def get_model_layers_unet(self):
        print("Getting UNET Layers")
        all_unet_modules = []
        for unets in self.quantizable_components["unet"]:
            unet_modules = []
            for name, module in getattr(self.pipeline, unets).named_children():
                unet_modules.append((name, module))
            all_unet_modules.append(unet_modules)
        return all_unet_modules
    
    def get_model_layers_te(self):
        print("Getting Text Encoder Layers")
        all_TE_modules = []
        for TE in self.quantizable_components["text_encoder"]:
            te_modules = []
            for name, module in getattr(self.pipeline, TE).named_children():
                te_modules.append((name, module))
            all_TE_modules.append(te_modules)
        return all_TE_modules
    
    def get_model_layers_vae(self):
        print("Getting VAE Layers")
        all_vae_modules = []
        for vae in self.quantizable_components["vae"]:
            vae_modules = []
            for name, module in getattr(self.pipeline, vae).decoder.named_children():
                vae_modules.append((name, module))
            all_vae_modules.append(vae_modules)
        return all_vae_modules
    
    def get_model_layers_transformers(self):
        raise Exception("There is no transformer in this model, StableDiffusion1_x, Ln69")

    def get_root(self, component, idx):
        return getattr(self.pipeline, self.quantizable_components[component][idx])

    def get_debugModuleNames(self, quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False):
        return []
    
    def get_scalingStates(self, quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False):
        return []

    def get_projectionNames(self, quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False):
        return []
    
    def get_components(self):
        return self.quantizable_components
    
    def get_unet(self):
        return self.pipeline.unet

    def set_quantized_components(self, component):
        self.quantized_components.append(component)
    
    def get_pipeline(self):
        return self.pipeline

    def get_smoothing_blocks(self):
        blocks = {}

        for name, submodule in self.pipeline.unet.named_modules():
            if isinstance(submodule, BasicTransformerBlock):
                blocks[name] = submodule
        return blocks
    
    def mean_of_dict(self, act_dict):
        #0: tensor, 1:tensor...
        all_tensors = list(act_dict.values())
        stacked_act = torch.stack(all_tensors)
        mean = torch.mean(stacked_act, dim = 0)
        #print("typical act shape: ", act_dict[0].shape)
        #print("Stacked_act: ", stacked_act.shape)
        #print("Mean: ", mean.shape)
        return mean

    
    def get_layers_for_scaling_unet(self, module: BasicTransformerBlock, hooks):
        layers = []
        layers.append(
            dict(
                prev_op=module.norm1,
                layers=[
                    module.attn1.to_q,
                    module.attn1.to_k,
                    module.attn1.to_v,
                ],

                activations_max= [self.mean_of_dict(hooks['attn1.to_q'].max_scales),
                                  self.mean_of_dict(hooks['attn1.to_k'].max_scales),
                                  self.mean_of_dict(hooks['attn1.to_v'].max_scales),  
                                 ]
            )
        )
        
        layers.append(
            dict(
                prev_op=module.norm3,
                layers=[module.ff.net[0].proj],
                activations_max= [self.mean_of_dict(hooks['ff.net.0.proj'].max_scales)]
            )
        )
        
        # if isinstance(module.ff.net[2], torch.nn.Linear):
        #     layers.append(
        #         dict(
        #             prev_op=module.ff.net[0].proj, # The GEGLU module which contains the linear layer
        #             layers=[module.ff.net[2]],
        #             activations_max= [self.mean_of_dict(hooks['ff.net.2'].max_scales)]
        #         )
        #     )

        return layers