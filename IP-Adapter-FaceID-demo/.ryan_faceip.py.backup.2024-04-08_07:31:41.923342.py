#GAME PLAN:
#    upon creating faceID adapter, it monkey-patches the pipeline. We gonna have to use FaceIP adapter for all of them unless we can undo its set_ip_adapter() function (which, should be possible...not sure what variable name is changed but we can revert it right?)
#    secondlyps subclass IPAdapterFaceID so that it doesn't modify the pipe right away, and maybe add an undo function as well? 
#TODO: Get this working and see if we can get a Steve in sd_previewer.ipynb

import rp
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import gradio as gr
import cv2
from glob import glob
from contextlib import contextmanager


import source.stable_diffusion as stable_diffusion
import source.stable_diffusion_labels as stable_diffusion_labels

@rp.CachedInstances
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

        self.unpatch() # Don't change anything right now

    def patch(self):
        self.sd.unet.set_attn_processor(self.ipadapter_attn_processors)

    def unpatch(self):
        self.sd.unet.set_attn_processor(self.original_attn_processors)

    def __enter__(self):
        self.patch()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unpatch()

@rp.memoized
def get_face_analysis_app():
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

@rp.memoized
def get_average_face_embedding(faces):
    assert all(rp.is_image  (face) for face in faces)

    #Make it as if it were straight out of cv2.imread
    face=[rp.as_rgb_image   (face) for face in faces]
    face=[rp.as_byte_image  (face) for face in faces]
    face=[rp.cv_bgr_rgb_swap(face) for face in faces]

    app=get_face_analysis_app()

    #Straight from the original inference code
    faceid_all_embeds = []
    for face in faces:
        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)

    return average_embedding
    

class IPAdapterLabel(stable_diffusion_labels.BaseLabel):
    def __init__(self, images, prompt="", negative_prompt=""):
        app=get_face_analysis_app()

    pass


def generate_image(prompt, negative_prompt):
#THIS IS JUST HERE FOR REFERENCE
    # sd = stable_diffusion._get_stable_diffusion_singleton()
    # pipe=sd.pipe

    # pipe.to(device)
    # app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))

    # faceid_all_embeds = []
    # for image in glob('./*.jpg'):
    #     face = cv2.imread(image)
    #     faces = app.get(face)
    #     faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    #     faceid_all_embeds.append(faceid_embed)
    
    # average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)

    patcher = IPAdapterPatcher()
    patcher.patch()
    ip_model=patcher.ip_model

    average_embedding
    
    image = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding, width=512, height=512, num_inference_steps=30
    )
    print(image)
    return image
