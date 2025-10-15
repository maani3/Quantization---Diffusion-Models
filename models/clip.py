import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.clip.modeling_clip import (
    CLIPModel as CLIPForConditionalGeneration,
    CLIPEncoderLayer as OldLlamaDecoderLayer
)
from awq.modules.fused.norm import FasterTransformerRMSNorm

class CLIPAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "CLIPEncoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: CLIPForConditionalGeneration):
        fuser = CLIPFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def checkQuantStatus( quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False ):
        # No checks for CLIP since it has all 4 components which can be chosen to be quantized
        return True

    @staticmethod
    def get_debugModuleNames( quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False ):
        moduleNames = []
        if quantVision == True:
            moduleNames.append( 'Vision ' )
        if quantVisionProjection == True:
            moduleNames.append( 'Vision Projection ' )
        if quantText == True:
            moduleNames.append( 'Text ' )
        if quantTextProjection == True:
            moduleNames.append( 'Text Projection ' )
        return moduleNames

    @staticmethod
    def get_scalingStates( quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False ):
        scalingStates = []
        if quantVision == True:
            scalingStates.append( True )
        if quantVisionProjection == True:
            scalingStates.append( False )
        if quantText == True:
            scalingStates.append( True )
        if quantTextProjection == True:
            scalingStates.append( False )
        return scalingStates
        
    @staticmethod
    def get_projectionNames( quantVision=False, quantText=True,  quantVisionProjection=False, quantTextProjection=False ):
        projectionNames = []
        if quantVision == True:
            projectionNames.append( '' )
        if quantVisionProjection == True:
            projectionNames.append( 'visual_projection' )
        if quantText == True:
            projectionNames.append( '' )
        if quantTextProjection == True:
            projectionNames.append( 'text_projection' )
        return projectionNames
        
    @staticmethod
    def get_toTensor():
        return ['attention_mask', 'causal_attention_mask','position_ids','inputs_embeds', 'text_embeds_projection', 'input_ids','image_embeds', 'image_embeds_projection', 'pixel_values', 'cache_position']

    @staticmethod
    def get_model_layers(model: CLIPForConditionalGeneration):
        # add final_layer_norm ??
        return model.text_model.encoder.layers

    @staticmethod
    def get_model_layers_textProjection(model: CLIPForConditionalGeneration):
        return [model.text_projection]

    @staticmethod
    def get_model_layers_vision(model: CLIPForConditionalGeneration):
        return model.vision_model.encoder.layers

    @staticmethod
    def get_model_layers_visionProjection(model: CLIPForConditionalGeneration):
        return [model.visual_projection]

    @staticmethod
    def get_act_for_scaling(module: OldLlamaDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: CLIPForConditionalGeneration, device: str):
        model.text_model.encoder.embed_tokens = model.text_model.embeddings.token_embedding.to(
            device
        )
        if hasattr(model.text_model.encoder, "rotary_emb"):
            model.text_model.encoder.rotary_emb = model.text_model.encoder.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling_SQ(module: OldLlamaDecoderLayer, act_scales):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.layer_norm1,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                input_scales=act_scales["self_attn.q_proj"],
                # module2inspect=module.self_attn,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        # if module.self_attn.v_proj.weight.shape == module.self_attn.out_proj.weight.shape:
        #     layers.append(
        #         dict(
        #             prev_op=module.self_attn.v_proj,
        #             layers=[module.self_attn.out_proj],
        #             inp=input_feat["self_attn.out_proj"],
        #         )
        #     )

        # linear 1
        layers.append(
            dict(
                prev_op=module.layer_norm2,
                layers=[module.mlp.fc1],
                input_scales=act_scales["mlp.fc1"],
                # module2inspect=module.mlp,
            )
        )

        # # linear 2
        # layers.append(
        #     dict(
        #         prev_op=module.mlp.fc1,
        #         layers=[module.mlp.fc2],
        #         inp=input_feat["mlp.fc2"],
        #     )
        # )

        return layers

    @staticmethod
    def get_layers_for_scaling_vision_SQ(module: OldLlamaDecoderLayer, act_scales):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.layer_norm1,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                input_scales=act_scales["self_attn.q_proj"],
                # module2inspect=module.self_attn,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        # if module.self_attn.v_proj.weight.shape == module.self_attn.out_proj.weight.shape:
        #     layers.append(
        #         dict(
        #             prev_op=module.self_attn.v_proj,
        #             layers=[module.self_attn.out_proj],
        #             inp=input_feat["self_attn.out_proj"],
        #         )
        #     )

        # linear 1
        layers.append(
            dict(
                prev_op=module.layer_norm2,
                layers=[module.mlp.fc1],
                input_scales=act_scales["mlp.fc1"],
                # module2inspect=module.mlp,
            )
        )

        # # linear 2
        # layers.append(
        #     dict(
        #         prev_op=module.mlp.fc1,
        #         layers=[module.mlp.fc2],
        #         inp=input_feat["mlp.fc2"],
        #     )
        # )

        return layers
        
    @staticmethod
    def get_layers_for_scaling(module: OldLlamaDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.layer_norm1,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.out_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.out_proj],
                    inp=input_feat["self_attn.out_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.layer_norm2,
                layers=[module.mlp.fc1],
                inp=input_feat["mlp.fc1"],
                module2inspect=module.mlp,
            )
        )

        # linear 2 -- this results in loss -- wHY??
        # layers.append(
        #     dict(
        #         prev_op=module.mlp.fc1,
        #         layers=[module.mlp.fc2],
        #         inp=input_feat["mlp.fc2"],
        #     )
        # )

        return layers

    @staticmethod
    def get_layers_for_scaling_vision(module: OldLlamaDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.layer_norm1,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.out_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.out_proj],
                    inp=input_feat["self_attn.out_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.layer_norm2,
                layers=[module.mlp.fc1],
                inp=input_feat["mlp.fc1"],
                module2inspect=module.mlp,
            )
        )

        # linear 2 -- this results in loss -- wHY??
        # layers.append(
        #     dict(
        #         prev_op=module.mlp.fc1,
        #         layers=[module.mlp.fc2],
        #         inp=input_feat["mlp.fc2"],
        #     )
        # )

        return layers

class CLIPFuser:
    def __init__(self, model: CLIPForConditionalGeneration):
        self.model = model.language_model

        self.llama_blocks: List[Tuple[str, OldLlamaDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "LlamaDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldLlamaDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon,
            )
            if hasattr(self.model.config, "max_seq_len"):
                max_seq_len = self.model.config.max_seq_len
            else:
                max_seq_len = self.model.config.max_position_embeddings
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
