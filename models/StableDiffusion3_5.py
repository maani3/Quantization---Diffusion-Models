from .base import BaseAWQForDiffusion
from .base import QUANTISABLE_COMPONENTS

class StableDiffusion3_5(BaseAWQForDiffusion):

    def __init__(self, pipeline, model_type, is_quantized, config, quant_config, refiner_path = None, access_token = "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"):
        if refiner_path is not None:
            raise Exception("StableDiffusion3.5 has no refiner model, if there is its not supported")
    
        super().__init__(pipeline, model_type, is_quantized, config, quant_config)
        print("A StableDiffusion3.5 model has been created!")
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
        if quantUnet == True:
            raise Exception("There is no UNET in StableDiffusion3_5")

    def get_model_layers_transformers(self):
        print("Getting Transformer Layers")
        all_transformer_modules = []
        for transformer in self.quantizable_components["transformer"]:
            transformer_modules = []
            for name, module in getattr(self.pipeline, transformer).named_children():
                transformer_modules.append((name, module))
            all_transformer_modules.append(transformer_modules)
        return all_transformer_modules
    
    def get_model_layers_unet(self):
        raise Exception("NO UNET IN THIS MODEL")
    
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
    
    def get_transformer(self):
        return self.pipeline.transformer

    def set_quantized_components(self, component):
        self.quantized_components.append(component)
