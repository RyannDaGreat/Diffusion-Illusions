#GAME PLAN:
#    upon creating faceID adapter, it monkey-patches the pipeline. We gonna have to use FaceIP adapter for all of them unless we can undo its set_ip_adapter() function (which, should be possible...not sure what variable name is changed but we can revert it right?)
#    secondly, it replaces the prompt embeddings with longer ones, concatt'd with the image prompt embeddings...
#TODO:
#    monitor changes made by 
#    Make IPAdapterFaceID label objects that require an IPAdapter instance and FaceAnalysis object (used here) and list of face images
#       (as rp images). It stores a private _average_embedding object as seen in here, but thats only an intermediate value  -  it functions as a normal
#    Ideally, pipe could be dynamically patched and unpatched when it detects its using a label....
#       BUT...we can probably just use a WITH block instead or something...
#    We need IPAdapterFaceID to be able to patch the pipe in our StableDiffusion pipeline
#       Just keep track of pipe.unet.attn_processors  -  this is what gets changed and it can be reverted.
#           Perhaps subclass IPAdapterFaceID so that it doesn't modify the pipe right away, and maybe add an undo function as well? 

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import gradio as gr
import cv2
from glob import glob

import source.stable_diffusion as stable_diffusion
import source.stable_diffusion_labels as stable_diffusion_labels

class IPAdapterPatcher:

    def __init__(self):
        self.sd = stable_diffusion._get_stable_diffusion_singleton()
        self.base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        self.ip_ckpt = hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID",
            filename="ip-adapter-faceid_sd15.bin",
            repo_type="model",
        )

        self.device=self.sd.device


        self.original_attn_processors = self.sd.unet.attn_processors
        self.ip_model = IPAdapterFaceID(self.sd.pipe, self.ip_ckpt, self.device)
        self.ipadapter_attn_processors = self.sd.unet.attn_processors

        self.unset_attns() # Don't change anything right now

    def unset_attns(self):
        self.sd.unet.set_attn_processor(self.original_attn_processors)

    def set_attns(self):
        self.sd.unet.set_attn_processor(self.original_attn_processors)


class IPAdapterLabel(stable_diffusion_labels.BaseLabel):
    pass


def generate_image(prompt, negative_prompt):
    pipe.to(device)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faceid_all_embeds = []
    for image in glob('./*.jpg'):
        face = cv2.imread(image)
        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)
    
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
    
    image = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding, width=512, height=512, num_inference_steps=30
    )
    print(image)
    return image
