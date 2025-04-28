from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

class FlipIllusion(nn.Module):

    def __init__(self, checkpoint_path, device='mps', pipe=None):
        super().__init__()
        self.device = torch.device(device)
        
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint_path,
                scheduler=DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler"),
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
        
        if isinstance(prompts,str):
            prompts = [prompts]
    
        text_input = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length, 
            truncation=True, return_tensors='pt').input_ids.to(self.device)
        
        uncond_input = self.tokenizer(
            [self.uncond_text] * len(prompts), 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt').input_ids.to(self.device)
    
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.to(self.device))[0]
            uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]
    
        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
        return output_embeddings
    
    def encode_image(self, image):
        """ Assume image is a CHW torch tensor with values between 0 and 1 """
        image = rp.as_torch_image(image, device=self.device, dtype=torch.float32) #If given a numpy image or PIL image, convert it. Else, if given torch image, leaves it alone.
        image = 2 * image - 1  
        image = image.to(device=self.device)
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
        latents = 0.18215 * latents
        return latents
    
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
    
    def create_latents(self, height=512, width=512):
        """
        First create regular latent
        """
        latent_shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        unflip_latent = torch.randn(latent_shape, device=self.device)
        flip_latent = self.flip_latent(unflip_latent) #YOU CAN'T DO THAT - This is noise!!! 
        
        h = latent_shape[2]  
        sharpness = 50.0  
        normalized_positions = torch.linspace(-1, 1, h).to(self.device)
        gradient_mask = torch.sigmoid(sharpness * -normalized_positions).view(1, 1, h, 1).repeat(1, latent_shape[1], 1, latent_shape[3])
        final_latent = unflip_latent * gradient_mask + flip_latent * (1 - gradient_mask)
        return unflip_latent, flip_latent, final_latent, gradient_mask

    @torch.no_grad()
    def get_images(self, latents: list):
        images = []
        for latent in latents:
            latent_image = latent / 0.18215
            image = self.vae.decode(latent_image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image * 255).round().to(torch.uint8)
            image = image.permute(0, 2, 3, 1).squeeze(0)
            images.append(Image.fromarray(image.cpu().numpy()))
        return tuple(images)
    
    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5):

        text_embeddings = []
        for prompt in prompts:
            embedding = self.get_text_embedding(prompt)
            text_embeddings.append(embedding)

        latents = self.create_latents()
        unflip_latent, flip_latent, final_latent, gradient_mask = latents

        # Sanity Check
        print(f"Unflip Latent Shape: {unflip_latent.shape}")
        print(f"Flip Latent Shape: {flip_latent.shape}")

        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # Process the first prompt (unflipped)
            latent_model_input_1 = torch.cat([unflip_latent] * 2) if guidance_scale > 1.0 else unflip_latent
            latent_model_input_1 = self.scheduler.scale_model_input(latent_model_input_1, t)
            
            # Get noise prediction for first prompt
            with torch.no_grad():
                noise_pred_1 = self.unet(
                    latent_model_input_1, t, encoder_hidden_states=text_embeddings[0]
                ).sample
            
            # Process the second prompt (flipped latent)
            latent_model_input_2 = torch.cat([flip_latent] * 2) 
            latent_model_input_2 = self.scheduler.scale_model_input(latent_model_input_2, t)
            
            # Get noise prediction for second prompt
            with torch.no_grad():
                noise_pred_2 = self.unet(
                    latent_model_input_2, t, encoder_hidden_states=text_embeddings[1]
                ).sample
                
            # Flip the noise prediction for the second prompt
            noise_pred_2_flipped = self.flip_latent(noise_pred_2)
            
            unflip_noise_uncond, unflip_noise_cond = noise_pred_1.chunk(2)
            flip_noise_uncond_f , flip_noise_cond_f  = noise_pred_2_flipped.chunk(2)
            
            # Combine noise predictions using gradient mask
            combined_noise_pred = noise_pred_1 * gradient_mask[0:1] + noise_pred_2 * (1 - gradient_mask[0:1])

            # Get Noise Predict for All
            unflip_noise_pred = unflip_noise_uncond + guidance_scale * (unflip_noise_cond - unflip_noise_uncond)
            flip_noise_pred = flip_noise_uncond_f + guidance_scale * (flip_noise_cond_f - flip_noise_uncond_f)
            avg_noise_pred = (unflip_noise_pred + flip_noise_pred) * 0.5

            # Update individual latents
            unflip_latent = self.scheduler.step(unflip_noise_pred, t, unflip_latent).prev_sample
            flip_latent = self.scheduler.step(flip_noise_pred, t, flip_latent).prev_sample
            final_latent = self.scheduler.step(unflip_noise_pred , t, final_latent).prev_sample

        all_images = self.get_images([unflip_latent, flip_latent, final_latent])
        return all_images
    

if __name__ == '__main__':
    diffusion = FlipIllusion(checkpoint_path="stable-diffusion-v1-5/stable-diffusion-v1-5")

    prompts = [
        "Oil painting of Golden Retriever",
        "Oil painting of Golden Retriever"
    ]

    images = diffusion.sample(prompts)
    for i, image in enumerate(images):
        file = f"image_{i}.png"
        image.save(file)
