from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


class FlipIllusion(nn.Module):
    def __init__(self, checkpoint_path, device="mps", pipe=None):
        super().__init__()
        self.device = torch.device(device)

        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint_path,
                scheduler=DDIMScheduler.from_pretrained(
                    checkpoint_path, subfolder="scheduler"
                ),
                torch_dtype=torch.float,
                use_safetensors=True,
                requires_safety_checker=False,
                safety_checker=None,
            )

        self.pipe = pipe
        self.vae = pipe.vae.to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet = pipe.unet.to(self.device)
        self.scheduler = pipe.scheduler

        self.uncond_text = ""
        self.checkpoint_path = checkpoint_path

    def get_text_embedding(self, prompts, negative_prompt=None):
        """
        Embed Fucking text into Token Space
        """

        if isinstance(prompts, str):
            prompts = [prompts]

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        uncond_input = self.tokenizer(
            [self.uncond_text] * len(prompts),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.to(self.device))[0]
            uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]

        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return output_embeddings

    @torch.no_grad
    def encode_image(self, image):
        """ Takes in 3HW torch tensor with values between 0 and 1, returns latent CHW tensor """
        image = rp.as_torch_image(
            image, device=self.device, dtype=torch.float32
        )  # If given a numpy image or PIL image, convert it. Else, if given torch image, leaves it alone.
        
        image = 2 * image - 1
        image = image.to(device=self.device)
        latents = self.vae.encode(image).latent_dist.sample()
        latents = 0.18215 * latents
        return latents

    @torch.no_grad()
    def decode_latent(self, latent):
        """ Takes in latent CHW torch tensor, returns 3HW torch tensor with values between 0 and 1 """
        
        latent = latent[None] #CHW -> 1CHW
        
        latent = latent / 0.18215
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = image[0] #1CHW -> CHW
        
        return image

    @torch.no_grad()
    def decode_latents(self, latents: list):
        """ Takes in list of latent CHW torch tensors, returns list of 3HW torch tensors with values between 0 and 1 """
        return [self.decode_latent(x) for x in latents]

    @torch.no_grad()
    def flip_latent(self, latent):
        """
        Decode latent to image, flip it, and re-encode to latent
        """
        latent_scaled = latent / 0.18215
        image = self.vae.decode(latent_scaled).sample
        flipped_image = torch.flip(image, dims=[2, 3])
        flipped_latent = self.vae.encode(flipped_image).latent_dist.sample()
        flipped_latent = 0.18215 * flipped_latent
        return flipped_latent

    @torch.no_grad
    def pred_noise(self, latent, t, guidance_scale, text_embedding):
        latent=latent[None] #CHW -> 1CHW
        
        model_input = torch.cat([latent] * 2)
        model_input = self.scheduler.scale_model_input(model_input, t)

        noise_pred = self.unet(
            model_input, t, encoder_hidden_states=text_embedding
        ).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        
        noise_pred=noise_pred[0] #1CHW -> CHW

        return noise_pred

    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5):

        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        for t in tqdm(timesteps, total=len(timesteps)):
            for i, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):
                noise_pred = self.pred_noise(latent, t, guidance_scale, text_embeddings[0])
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i]=latent

        all_images = self.decode_latents(latents)
        
        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images


if __name__ == "__main__":
    diffusion = FlipIllusion(
        checkpoint_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )

    prompts = ["Oil painting of Golden Retriever", "Oil painting of Golden Retriever"]

    images = diffusion.sample(prompts)
    for i, image in enumerate(images):
        file = f"image_{i}.png"
        fansi_print(f'SAVED: {save_image(image,file)}', 'green bold')
