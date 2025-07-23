from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


def add_noise(
    clean_sample,
    noise,
    timestep,
    alphas_cumprod,
):
    # clean_sample is x_0 [aka original_sample].
    # Returns x_T
    # Larger alpha_prod --> less noise added  (more alpha --> less noise)
    # Earlier --> larger alpha_prod

    alpha_prod = alphas_cumprod[timestep]
    beta_prod  = 1 - alpha_prod
    sqrt_alpha_prod = alpha_prod**0.5
    sqrt_beta_prod  = beta_prod **0.5

    noisy_samples = sqrt_alpha_prod * clean_sample + sqrt_beta_prod * noise
    return noisy_samples


def get_velocity(
    sample,
    noise,
    timestep,
    alphas_cumprod,
):
    
    alpha_prod = alphas_cumprod[timestep]
    beta_prod  = 1 - alpha_prod
    sqrt_alpha_prod = alpha_prod**0.5
    sqrt_beta_prod  = beta_prod **0.5

    velocity = sqrt_alpha_prod * noise - sqrt_beta_prod * sample
    return velocity


def get_epsilon(
    sample,
    model_output,
    timestep,
    alphas_cumprod,
    pred_type,
):
    # Given the model might be trained on different objectives (predicting noise, clean samples, or v_prediction)
    # this function extracts the noise implicitly preidcted by it
    
    alpha_prod = alphas_cumprod[timestep]
    beta_prod  = 1 - alpha_prod
    sqrt_alpha_prod = alpha_prod**0.5
    sqrt_beta_prod  = beta_prod **0.5

    if pred_type == "epsilon":
        return model_output

    elif pred_type == "sample":
        return (sample - sqrt_alpha_prod * model_output) / sqrt_beta_prod

    elif pred_type == "v_prediction":
        return sqrt_alpha_prod * model_output + sqrt_beta_prod * sample
        return add_noise(clean_sample=model_output, noise=sample, timestep=timestep)  # Weird but equivalent


def get_clean_sample(
    sample,
    model_output,
    timestep,
    alphas_cumprod,
    pred_type,
):
    
    alpha_prod = alphas_cumprod[timestep]
    beta_prod  = 1 - alpha_prod
    sqrt_alpha_prod = alpha_prod**0.5
    sqrt_beta_prod  = beta_prod **0.5

    if pred_type == "epsilon":
        return (sample - sqrt_beta_prod * model_output) / sqrt_alpha_prod

    elif pred_type == "sample":
        return sample

    elif pred_type == "v_prediction":
        return sqrt_alpha_prod * sample - sqrt_beta_prod * model_output

    if pred_type == "epsilon":
        pred_original_sample = (sample - sqrt_beta_prod * model_output) / alpha_prod ** 0.5
        pred_epsilon         = model_output
    elif pred_type == "sample":
        pred_original_sample = model_output
        pred_epsilon         = (sample - alpha_prod ** 0.5 * pred_original_sample) / sqrt_beta_prod
    elif pred_type == "v_prediction":
        pred_original_sample = (alpha_prod**0.5) * sample - (beta_prod**0.5) * model_output
        pred_epsilon         = (alpha_prod**0.5) * model_output + (beta_prod**0.5) * sample

    return pred_epsilon, pred_original_sample


if not 'get_pipeline' in vars():
    @memoized
    def get_pipeline(checkpoint_path):
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
        return pipe


class RyanDiffusion(nn.Module):
    def __init__(self, checkpoint_path, device="mps", pipe=None):
        super().__init__()
        self.device = torch.device(device)

        if pipe is None:
            pipe = get_pipeline(checkpoint_path)

        self.pipe         = pipe
        self.vae          = pipe.vae.to(self.device)
        self.tokenizer    = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet         = pipe.unet.to(self.device)
        self.scheduler    = pipe.scheduler

        self.uncond_text = ""
        self.checkpoint_path = checkpoint_path
        
    @property
    def timesteps(self):
        return self.scheduler.timesteps
    
    @property
    def alphas_cumprod(self):
        return self.scheduler.alphas_cumprod

    @torch.no_grad
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

        text_embeddings   = self.text_encoder(text_input.to(self.device))[0]
        uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]

        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return output_embeddings

    @torch.no_grad
    def encode_image(self, image):
        """ Takes in 3HW torch tensor with values between 0 and 1, returns latent CHW tensor """

        image = rp.as_torch_image(
            image, device=self.device, dtype=torch.float32
        )  # If given a numpy image or PIL image, convert it. Else, if given torch image, leaves it alone.

        image = image[None] # CHW -> 1CHW
        
        image = 2 * image - 1
        image = image.to(device=self.device)

        latents = self.vae.encode(image).latent_dist.sample()
        latents = 0.18215 * latents

        latents = latents[0] #1CHW -> CHW

        return latents

    @torch.no_grad
    def decode_latent(self, latent):
        """ Takes in latent CHW torch tensor, returns 3HW torch tensor with values between 0 and 1 """
        
        latent = latent[None] #CHW -> 1CHW

        latent = latent / 0.18215
        
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = image[0] #1CHW -> CHW
        
        return image

    def decode_latents(self, latents: list):
        """ Takes in list of latent CHW torch tensors, returns list of 3HW torch tensors with values between 0 and 1 """
        return [self.decode_latent(x) for x in latents]

    def pred_noise(self, latent, t, guidance_scale, text_embedding):
        latent=latent[None] #CHW -> 1CHW
        
        model_input = torch.cat([latent] * 2)
        model_input = self.scheduler.scale_model_input(model_input, t)

        noise_pred = self.unet(
            model_input, t, encoder_hidden_states=text_embedding,
        ).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_guidance = noise_pred_cond - noise_pred_uncond
        noise_pred = noise_pred_uncond + guidance_scale * noise_pred_guidance
        
        noise_pred=noise_pred[0] #1CHW -> CHW

        return noise_pred

    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5):
        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)
        alphas_cumprod = self.scheduler.alphas_cumprod
        timesteps = self.scheduler.timesteps

        for t in tqdm(self.timesteps, total=len(self.timesteps)):
            for i, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):
                noise_pred = self.pred_noise(latent, t, guidance_scale, text_embedding)
                
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i]=latent

        all_images = self.decode_latents(latents)
        
        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images

class Illusion(RyanDiffusion):
    def apply_image_func_to_clean_latent(self, func, clean_latent):
        """ 
        Apply an image function to a non-noisy latent (func is a function operating on a 3x512x512 torch tensor with values between 0 and 1)
        """
        image = self.decode_latent(clean_latent)
        modified_image = func(image)
        modified_latent = self.encode_image(modified_image)
        return modified_latent

    def apply_image_func_to_noisy_latent(self, func, noisy_latent, noise_pred, t):
        """ 
        Apply an image function (func is a function operating on a 3x512x512 torch tensor with values between 0 and 1)
        to a noisy latent, gracefully - only applying it to the clean portion
        """
        clean_pred = get_clean_sample(
            sample         = noisy_latent,
            model_output   = noise_pred,
            pred_type      = "epsilon",
            timestep       = t,
            alphas_cumprod = self.alphas_cumprod,
        )

        modified_clean_pred = self.apply_image_func_to_clean_latent(func, clean_pred)

        modified_noisy_latent = get_epsilon(
            sample         = noisy_latent,
            model_output   = modified_clean_pred,
            pred_type      = "sample",
            timestep       = t,
            alphas_cumprod = self.alphas_cumprod,
        )

        return modified_noisy_latent


class ImageFilterDiffusion(Illusion):

    def image_filter(self, image):
        #Basic filter - should be overridden
        return image

    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5):
        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)

        for t in tqdm(self.timesteps, total=len(self.timesteps)):
            for i, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):
                noise_pred = self.pred_noise(latent, t, guidance_scale, text_embedding)

                noise_pred = self.apply_image_func_to_noisy_latent(
                    self.image_filter,
                    latent,
                    noise_pred,
                    t,
                )

                #UNCOMMENT TO VIEW PROGRESS AS IT DIFFUSES! PRETTY COOL TO SEE!
                clean_pred = get_clean_sample(
                    sample         = latent,
                    model_output   = noise_pred,
                    pred_type      = "epsilon",
                    timestep       = t,
                    alphas_cumprod = self.alphas_cumprod,
                )
                display_image(self.decode_latent(clean_pred))
                
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i] = latent

        all_images = self.decode_latents(latents)
        
        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images

class GrayscaleFilterDiffusion(ImageFilterDiffusion):
    def image_filter(self, image):
        """ A grayscale image filter for a 3HW tensor """
        return image.mean(0,keepdim=True).repeat(3,1,1)


class PixelArt(Illusion):

    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5):
        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)

        for t in tqdm(self.timesteps, total=len(self.timesteps)):
            for i, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):
                noise_pred = self.pred_noise(latent, t, guidance_scale, text_embedding)
                
                def rgb_to_grayscale(image):
                    return image.mean(0,keepdim=True).repeat(3,1,1)

                noise_pred = self.apply_image_func_to_noisy_latent(
                    rgb_to_grayscale,
                    latent,
                    noise_pred,
                    t,
                )

                # clean_pred = get_clean_sample(
                #     sample         = latent,
                #     model_output   = noise_pred,
                #     pred_type      = "epsilon",
                #     timestep       = t,
                #     alphas_cumprod = self.alphas_cumprod,
                # )
                # display_image(self.decode_latent(clean_pred))
                
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i] = latent

        all_images = self.decode_latents(latents)
        
        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images

if __name__ == "__main__":
    torch.manual_seed(42)
    
    diffusion = GrayscaleFilterDiffusion(
        checkpoint_path="stable-diffusion-v1-5/stable-diffusion-v1-5"
    )

    prompts = [
        # "Oil painting of Golden Retriever",
        "Oil painting of Golden Retriever",
    ]

    images = diffusion.sample(prompts)
    for i, image in enumerate(images):
        file = f"image_{i}.png"
        fansi_print(f'SAVED: {save_image(image,file)}', 'green bold')
