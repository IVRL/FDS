from typing import Tuple, Union, Optional, List
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
import pillow_avif

# More descriptive type aliases
Tensor = torch.Tensor
OptionalTensor = Optional[Tensor]
TensorSequence = Union[Tuple[Tensor, ...], List[Tensor]]

def load_512(image_path: str, left=0, right=0, top=0, bottom=0) -> np.ndarray:
    """
    Load an image, convert to RGB, and resize to 512x512.
    
    Args:
        image_path: Path to the input image
        left, right, top, bottom: Crop parameters (currently unused)
        
    Returns:
        numpy array of shape (512, 512, 3) containing the RGB image
    """
    image = np.array(Image.open(image_path).convert("RGB").resize((512, 512)))[:, :, :3]  
    return image

@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str, device: torch.device = torch.device('cuda:0')) -> Tensor:
    """
    Generate text embeddings using the Stable Diffusion pipeline's text encoder.
    
    Args:
        pipe: Stable Diffusion pipeline
        text: Input text to encode
        device: The device to run the encoding on
        
    Returns:
        Tensor containing the text embeddings
    """
    tokens = pipe.tokenizer(
        [text], 
        padding="max_length", 
        max_length=77, 
        truncation=True,
        return_tensors="pt", 
        return_overflowing_tokens=True
    ).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image: Tensor) -> np.ndarray:
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]

@torch.no_grad()
def decode(latent: Tensor, pipe: StableDiffusionPipeline, im_cat: OptionalTensor = None) -> Image.Image:
    image = pipe.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)

def init_pipe(device: torch.device, dtype: torch.dtype, unet: UNet2DConditionModel, scheduler) -> Tuple[UNet2DConditionModel, Tensor, Tensor]:
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    for p in unet.parameters():
        p.requires_grad = False
    return unet, alphas, sigmas

