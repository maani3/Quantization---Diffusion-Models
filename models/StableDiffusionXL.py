from .base import BaseAWQForDiffusion, DiffusionPipeline
from .base import QUANTISABLE_COMPONENTS

class StableDiffusionXL(BaseAWQForDiffusion):

    def __init__(self, base_pipeline,model_type, is_quantized, config, quant_config, refiner_path = None, access_token = None):
        super().__init__(base_pipeline, model_type, is_quantized, config, quant_config)
        print("A StableDiffusionXL model has been created!")
        self.quantizable_components = {"unet": [], "text_encoder":[], "vae":[], "transformer":[]}
        self.quantized_components = []
        self.token = access_token

        if refiner_path is not None:
            self.refiner_pipeline = DiffusionPipeline.from_pretrained(refiner_path, torch_dtype = torch.float16)
            print("Refiner Model Loaded as well!")
        else:
            self.refiner_pipeline = None
            
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
        raise Exception("There is no transformer in this model, StableDiffusionXL, Ln66")

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
    
    def get_quantized_components(self):
        return self.quantizable_components