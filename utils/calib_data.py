import torch
import logging
from awq.models.base import diffusers
from typing import List, Union, Dict, Any
from typing_extensions import Doc, Annotated
from tqdm import tqdm
from datasets import load_dataset
from diffusers.callbacks import PipelineCallback


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]


def custom_multimodal_dataset(
    data_dict: dict,
):
    return data_dict

class CalibrationCallback(PipelineCallback):
    def __init__(self):
        super().__init__()
        self.tensor = ["prompt_embeds"]
        #self.checkpt = 5  Useful for Q-diff/Blockwise Reconsutrction
        self.prompt_embeds = {}
        self.batch_num = 0
        self.prompt_embeds[self.batch_num] = []

    @property
    def tensor_inputs(self):
        return self.tensor
    
    def set_batch_num(self, batch_no):
        self.batch_num = batch_no
        self.prompt_embeds[self.batch_num] = []

    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) -> Dict[str, Any]:
        prompt_embeds = callback_kwargs["prompt_embeds"]

        if len(self.prompt_embeds[self.batch_num])==0:
            self.prompt_embeds[self.batch_num] = prompt_embeds
            
        return self.prompt_embeds

    def __call__(self, pipeline, step_index, timestep, callback_kwargs: dict):
        return self.callback_fn(pipeline, step_index, timestep, callback_kwargs)

class Mean_Max_Activation_Hook:

    def __init__(self):
        self.hook_handle = None
        self.max_scales = {}
        self.step = 0

    def __call__(self, module, module_in, module_out):

        #self.inputs[self.step] = module_in[0]
        #Incoming shape is either [batch_size, seq_len, channels] or [batch_size, channels]
        n_channels = module_in[0].shape[-1]
        max_per_channel = module_in[0].reshape([-1, n_channels]).abs().amax(dim = 0)
        #mean_per_channel = module_in[0].reshape([-1, n_channels]).abs().mean(dim = 0)
        self.max_scales[self.step]=max_per_channel
        #self.mean_scales[self.step] = mean_per_channel
        self.step = self.step + 1

    def clear(self):
        self.max_scales = []

class Layer_Norm_Hook:
    def __init__(self):
        self.hook_handle = None
        self.step = 0
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        self.outputs[self.step]=module_out
        self.step = self.step + 1
        
    def clear(self):
        self.outputs = []

def generate_latents(pipe, batch_size, device, generator):
    num_in_channels = pipe.unet.config.in_channels
    height = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[0]
        )
    width = (
            pipe.unet.config.sample_size
            if pipe._is_unet_config_sample_size_int
            else pipe.unet.config.sample_size[1]
        )
    height, width = height * pipe.vae_scale_factor, width * pipe.vae_scale_factor
        
    shape = (
        batch_size,
        num_in_channels,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )
        
    if pipe.text_encoder is not None:
        m_dtype = pipe.text_encoder.dtype
    elif pipe.unet is not None:
        m_dtype = pipe.unet.dtype

    latents = diffusers.utils.torch_utils.randn_tensor(shape, generator=generator, device = device, dtype = m_dtype)
    return latents

def get_calib_dataset_dm(
    model_pipeline: Annotated[diffusers.DiffusionPipeline, Doc("Pipeline of the original Model")],
    text_dataset:  Annotated[Union[List[str], str], Doc("List of prompts to be given to a diffusion Model")],
    batch_size: Annotated[int, Doc("The batch_size of calibration")] = 4,
    n_samples: Annotated[int, Doc("The number of prompts")] = 100,
    seed: Doc("The Seed value for reproducibility") = 42,
    device: Annotated[str, Doc("The device on which the model is currently")]="cuda",
    split: Doc("Split of Dataset(Train/Test)") = None,
    text_column: Doc("Name of text column in the dataset") = None,
    cut_off_captions: Doc("Length of prompts in no. of characters") = 200,
    ):

    calib_data = []
    assert n_samples % batch_size == 0, "The batch_size, doesnt divide the dataset, choose an appropriate batch_size"
    n_batches = n_samples//batch_size

    if isinstance(text_dataset, str):
        if text_dataset == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            prompts = dataset[text_column]
        else:
            dataset = load_dataset(text_dataset, split=split, streaming = True)
            prompts = []
            for i, items in enumerate(dataset):
                if i >= n_samples:
                    break
                prompts.append(items[text_column][0:cut_off_captions])

    else:
        prompts = text_dataset
    
    generator = torch.manual_seed(seed)
    
    for i in tqdm(range(n_batches), desc="Generating Calibration Data"):
        start_index = i * batch_size
        prompt_batch = prompts[start_index:start_index + batch_size]
        latents = generate_latents(model_pipeline, batch_size, device, generator)
        calib_data.append((prompt_batch, latents))

    return calib_data

#Apply Hooks to Linear Layers
def apply_hook(module: torch.nn.Module):
    hook_dictionary = {}
    for name,submod in module.named_modules():
        if isinstance(submod, torch.nn.Linear):
            hook = Mean_Max_Activation_Hook()
            hook.hook_handle = submod.register_forward_hook(hook)
            hook_dictionary[name] = hook
    
    return hook_dictionary

#Run Calibration on Model
def run_calibration(pipeline: diffusers.DiffusionPipeline,
                    samples: Doc("List of Tuples (prompts_batch, latents_batch)"),
                    callback: Doc("To store timestep data and UNET outputs") = None,
                    n_inference_steps: Doc("How many iterations UNET should do")=50,
                    cfg: Doc("Strength of Classifier-free-guidance") = 7.5
                    ):

                    print("Runinng Calibration")
                    pipeline.to("cuda")
                    for i, sample in enumerate(samples):
                        if callback is not None:
                            callback.set_batch_num(i)
                        outputs = pipeline(prompt = sample[0],
                                           latents = sample[1],
                                           callback_on_step_end = callback,
                                           num_inference_steps = n_inference_steps,
                                           guidance_scale = cfg,
                                           num_images_per_prompt = 1).images
                    print("Calibration Done!")

def remove_hooks(hook_d: Doc("Hook Dictionary including layer_name: Hook")):
    for k,v in hook_d:
        v.hook_handle.remove()
