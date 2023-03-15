import torch
import numpy as np
from typing import Union, List
from transformers import AutoProcessor, CLIPModel, CLIPProcessor
import rp

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_logits(image: Union[torch.Tensor, np.ndarray], prompts: Union[List[str], str]) -> Union[torch.Tensor, np.ndarray]:
    """
    Takes a torch image and a list of prompt strings and returns a vector of log-likelihoods.
    The gradients can be propogated back into the image.

    Can also take in a numpy image. In this case, it will return a numpy vector instead of a torch vector.

    Parameters:
        image (Union[torch.Tensor, np.ndarray]): An image in CHW format if it's a torch.Tensor or in HWC format if it's a numpy array.
        prompts (List[str]): A list of prompt strings.

    Returns:
        Union[torch.Tensor, np.ndarray]: Vector of log-likelihoods in the form of a torch.Tensor if the input is a torch.Tensor or a numpy array if the input is a numpy array.
    """
    
    if isinstance(prompts, str):
        prompts=[prompts]
    
    if rp.is_image(image):
        # This block adds compatiability for numpy images
        # If given a numpy image, it will output a numpy vector of logits
        
        device = "cuda" # In the future, perhaps be smarter about the device selection
        image = rp.as_rgb_image  (image)
        image = rp.as_float_image(image)
        image = rp.as_torch_image(image)
        image = image.to(device)
        
        output = get_clip_logits(image, prompts)
        output = rp.as_numpy_array(output)
        assert output.ndim == 1, "output is a vector"
        
        return output
        
    #Input assertions
    assert isinstance(image,torch.Tensor), 'image should be a torch.Tensor'
    assert image.ndim==3, 'image should be in CHW form'
    assert image.shape[0]==3, 'image should be rgb'
    assert image.min()>=0, 'image should have values between 0 and 1'
    assert image.max()<=1, 'image should have values between 0 and 1'
    assert prompts, 'must have at least one prompt'
    assert all(isinstance(x,str) for x in prompts), 'all prompts must be strings'

    #This stupid clip_processor converts our image into a PIL image in the middle of its pipeline
    #This destroys the gradients; so the rest of the function will be spent fixing that
    image_hwc = image.permute(1,2,0) # (H,W,3)
    inputs = clip_processor(text=list(prompts), images=image_hwc.detach().cpu(), return_tensors="pt", padding=True)

    #There is a specific mean and std this clip_model expects
    mean = torch.tensor(clip_processor.feature_extractor.image_mean).to(image.device) # [0.4815, 0.4578, 0.4082]
    std  = torch.tensor(clip_processor.feature_extractor.image_std ).to(image.device) # [0.2686, 0.2613, 0.2758]

    #Normalize the image the way the clip_processor does
    norm_image = image # (3,H,W)
    norm_image = rp.torch_resize_image(norm_image,(224,224)) # (3,224,224)
    norm_image = norm_image[None] # (1,3,224,224)
    norm_image = (norm_image - mean[None, :, None, None]) / std[None, :, None, None]
    norm_image = norm_image.type_as(inputs["pixel_values"])
    inputs["pixel_values"] = norm_image
    
    #Put all input tensors on the same device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key]=inputs[key].to(image.device)

    #Calculate image-text similarity score
    outputs = clip_model.to(image.device)(**inputs) # Move the clip_model to the device we need on the fly
    logits_per_image = outputs.logits_per_image  # The image-text similarity score
    
    assert logits_per_image.shape == (1, len(prompts),)
    
    output = logits_per_image[0]
    assert output.shape == (len(prompts),)
    
    return output