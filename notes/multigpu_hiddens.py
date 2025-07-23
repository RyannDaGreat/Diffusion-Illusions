# 2025-07-23 17:13:20.349332
import inspect
from functools import cached_property

import torch
import torch.nn as nn
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

import rp


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
    @rp.memoized
    def get_pipeline(checkpoint_path, device):
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
        pipe = pipe.to(device)

        rp.fansi_print(f'Init Pipe: device={device},  checkpoint_path={checkpoint_path}', 'green bold')

        return pipe


class Diffusion(nn.Module):
    def __init__(self, *, checkpoint_path="stable-diffusion-v1-5/stable-diffusion-v1-5", device=None):
        super().__init__()

        if device is None:
            device = rp.select_torch_device(prefer_used=True)

        pipe = get_pipeline(checkpoint_path, device)

        self.pipe         = pipe
        self.device       = pipe.device
        self.vae          = pipe.vae
        self.tokenizer    = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet         = pipe.unet
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
    def _encode_image(self, image):
        """ Takes in 3HW torch tensor with values between 0 and 1, returns latent CHW tensor """

        image = rp.as_torch_image(
            image, device=self.device, dtype=torch.float32
        )  # If given a numpy image or PIL image, convert it. Else, if given torch image, leaves it alone.

        image = image[None] # CHW -> 1CHW

        image = 2 * image - 1
        image = image.to(device=self.device)

        latent = self.vae.encode(image).latent_dist.sample()
        latent = 0.18215 * latent

        latent = latent[0] #1CHW -> CHW

        return latent

    def encode_image(self, image):
        latent = self._encode_image(image)
        latent = self.correct_encoding_order1(latent)
        return latent

    def encode_if_image(self, image):
        latent_num_channels = 4
        if rp.is_torch_tensor(image) and image.ndim==3 and image.shape[0]==latent_num_channels:
            latent = image #It is already an image latent
        else:
            latent = self.encode_image(image)
        return latent

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

    def encode_images(self, images: list):
        return [self.encode_image(x) for x in images]

    def correct_encoding_order1(self, latent):
        # print("CORRECTING")
        e1=latent
        d1=self.decode_latent(e1)
        e2=self._encode_image(d1)
        e0_linear = 2*e1 - e2
        return e0_linear

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


    @cached_property
    def inverse_scheduler(self):
        return DDIMInverseScheduler.from_pretrained(self.checkpoint_path, subfolder='scheduler')

    @torch.no_grad()
    def ddim_inversion(self, image_or_latent, num_steps: int = 50) -> torch.Tensor:
        # This function is NOT thread safe if used concurrently on the same GPU due to the TemporarilySetAttr
        latents = self.encode_if_image(image_or_latent)[None]

        with rp.TemporarilySetAttr(self.pipe, scheduler=self.inverse_scheduler):
            inv_latents, _ = self.pipe(
                prompt="",
                negative_prompt="",
                guidance_scale=1.0,
                height=latents.shape[1]*8,
                width=latents.shape[2]*8,
                output_type="latent",
                return_dict=False,
                num_inference_steps=num_steps,
                latents=latents,
            )

        inv_latent = inv_latents[0]
        return inv_latent

    @torch.no_grad()
    def edict_inversion(self, image_or_latent, num_steps: int = 20) -> torch.Tensor:
        """
        Performs EDICT inversion using methods internal to this class.
        Assumes self.pipe is a standard Diffusers pipeline.
        MADE WITH GEMINI

        TODO: Make this func more uniform with other funcs in this file

        NOTE: The number of inversion steps used should MATCH the number of forward steps you use later or it wont work right! I found that out empirically...not theoretical at all...
        """
        # Add these methods to your class
        import torch

        #SETTINGS
        self.mixing_coeff = 0.93
        self.leapfrog_steps = True
        prompt=""
        guidance_scale=1.001#eh i need it to trigger 2 latents cause im too lazy to debug the non-guidance case

        #Three functions that are only ever used in edict_inversion
        def noise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
            """EDICT mixing operation for the reverse (inversion) process."""
            y = (y - (1 - self.mixing_coeff) * x) / self.mixing_coeff
            x = (x - (1 - self.mixing_coeff) * y) / self.mixing_coeff
            return [x, y]

        def _get_alpha_and_beta(self, t: torch.Tensor):
            """Gets the cumulative alpha and beta for a given timestep t."""
            # Assumes self.pipe.scheduler exists and is a DDIMScheduler
            t_int = int(t)
            alpha_prod = self.pipe.scheduler.alphas_cumprod[t_int] if t_int >= 0 else self.pipe.scheduler.final_alpha_cumprod
            return alpha_prod, 1 - alpha_prod

        def noise_step(self, base: torch.Tensor, model_input: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor):
            """Performs one step of the reverse ODE solve for EDICT."""
            # Assumes self.pipe.scheduler exists
            prev_timestep = timestep - self.pipe.scheduler.config.num_train_timesteps / self.pipe.scheduler.num_inference_steps
            alpha_prod_t, beta_prod_t = _get_alpha_and_beta(self, timestep)
            alpha_prod_t_prev, beta_prod_t_prev = _get_alpha_and_beta(self, prev_timestep)

            a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
            b_t = -a_t * (beta_prod_t**0.5) + beta_prod_t_prev**0.5

            next_model_input = (base - b_t * model_output) / a_t
            return model_input, next_model_input.to(base.dtype)


        # 1. Setup
        device = self.pipe.device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode image into latents
        image_latents = self.encode_if_image(image_or_latent)[None]

        # 3. Encode prompt
        text_embeds = self.pipe._encode_prompt(
            prompt, device, 1, do_classifier_free_guidance
        )

        # 4. Prepare timesteps for inversion
        self.pipe.scheduler.set_timesteps(num_steps, device)
        timesteps = self.pipe.scheduler.timesteps.flip(0) # Invert from T to 0

        # 5. The EDICT inversion loop
        coupled_latents = [image_latents.clone(), image_latents.clone()]
        for i, t in enumerate(rp.eta(timesteps)):
            coupled_latents = noise_mixing_layer(self, x=coupled_latents[0], y=coupled_latents[1])

            # Leapfrog steps for stability
            for j in range(2):
                k = j ^ 1
                if self.leapfrog_steps and i % 2 == 0:
                    k, j = j, k
                
                model_input = coupled_latents[j]
                base = coupled_latents[k]

                # Predict the noise
                latent_model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeds).sample

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Perform the reverse EDICT step
                base, model_input = noise_step(self,
                    base=base, model_input=model_input, model_output=noise_pred, timestep=t,
                )
                coupled_latents[k] = model_input

        # Return one of the resulting coupled latents
        return coupled_latents[0][0]

    @torch.no_grad()
    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None, decode=True):
        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = latents if latents is not None else [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)

        for t in tqdm(self.timesteps):
            for i, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):
                noise_pred = self.pred_noise(latent, t, guidance_scale, text_embedding)

                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i]=latent

        if not decode:
            return latents

        all_images = self.decode_latents(latents)

        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images

def demo_ddim_inversion():
    diffusion=Diffusion()
    image=rp.load_image('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png')
    latent=diffusion.ddim_inversion(image)
    null_prompt_reconstructions = diffusion.sample(prompts=[''],                     latents=[latent], guidance_scale=3)
    reconstructions             = diffusion.sample(prompts=['anime woman in a hat'], latents=[latent], guidance_scale=3)
    rp.display_image(
        rp.horizontally_concatenated_images(
            image,
            null_prompt_reconstructions[0],
            reconstructions[0],
        )
    )

@rp.globalize_locals
def edict_inversion_demo():
    diffusion=Diffusion()
    image=load_image('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png')
    #image=load_image('/Users/ryan/Downloads/download.jpeg')
    #latent=diffusion.ddim_inversion(image,20)
    latent=diffusion.edict_inversion(image,"",20)
    null_prompt_reconstructions = diffusion.sample(prompts=[''],                     latents=[latent], guidance_scale=3)
    reconstructions             = diffusion.sample(prompts=['anime woman in a hat'], latents=[latent], guidance_scale=3)
    display_image(
        horizontally_concatenated_images(
            image,
            null_prompt_reconstructions[0],
            reconstructions[0],
        )
    )

class SeamlessGenerator(Diffusion):

    @staticmethod
    def roll_image(image, *, dx, dy):
        return image.roll(dy, 1).roll(dx, 2)

    @staticmethod
    def random_roll(image, *, do_x:bool, do_y:bool):
        C, H, W = image.shape
        dx = rp.random_index(W) * do_x
        dy = rp.random_index(H) * do_y
        return SeamlessGenerator.roll_image(image, dx=dx, dy=dy), (dx, dy)

    @torch.no_grad
    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None, do_x:bool=True, do_y:bool=True, decode=True):
        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = latents if latents is not None else [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)

        for t in tqdm(self.timesteps):
            for i, (latent, text_embedding) in enumerate(zip(latents, text_embeddings)):

                #Randomly shift the image
                latent, (dx, dy) = self.random_roll(latent, do_x=do_x, do_y=do_y)

                noise_pred = self.pred_noise(latent, t, guidance_scale, text_embedding)
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample

                #Put it back again
                latent = self.roll_image(latent, dx=-dx, dy=-dy)

                latents[i]=latent


        if not decode:
            return latents

        all_images = self.decode_latents(latents)

        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        #TODO: Decode the image with multiple shifts, and use laplacian blending to stitch them together
        #Becuase, right now most of the seams come from the decoder

        return all_images

    @staticmethod
    def demo():
        g=SeamlessGenerator()
        ans=g.sample(['wood texture'],do_y=True,do_x=True,num_steps=10)
        image=rp.grid_concatenated_images([list(ans)*3]*3)
        print(rp.save_image(image))
        rp.display_image(image)


class ImageFilterDiffusion(Diffusion):
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

    def image_filter(self, image):
        #Basic filter - should be overridden
        return image

    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None, decode=True):
        text_embeddings = [self.get_text_embedding(x) for x in prompts]

        latents = latents if latents is not None else [
            torch.randn(4, 64, 64).to(self.device, torch.float32) for x in prompts
        ]

        self.scheduler.set_timesteps(num_steps, device=self.device)

        for t in tqdm(self.timesteps):
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
                rp.display_image(self.decode_latent(clean_pred))

                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i] = latent

        if not decode:
            return latents

        all_images = self.decode_latents(latents)

        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images

class GrayscaleFilterDiffusion(ImageFilterDiffusion):
    def image_filter(self, image):
        """ A grayscale image filter for a 3HW tensor """
        return image.mean(0,keepdim=True).repeat(3,1,1)


class PixelArtDiffusion(ImageFilterDiffusion):
    def image_filter(self, image):
        """ A grayscale image filter for a 3HW tensor """
        #return image
        PIXEL_SIZE=16
        QUANT=8 #Number of colors per channel
        image = rp.torch_resize_image(image, 1/PIXEL_SIZE,interp='nearest')
        image = (image * QUANT).round()/QUANT
        image = rp.torch_resize_image(image, PIXEL_SIZE, interp='nearest')
        return image

class DiffusionIllusion:
    def __init__(self, diffusions: list[Diffusion], parallel=True):
        assert len(diffusions) == self.num_derived_images, f'Expect to have one Diffusion per target, but {len(diffusions)} != {self.num_derived_images}'

        self.diffusions = diffusions
        self.num_threads = len(diffusions) if parallel else 0

    @property
    def primary_diffusion(self):
        return self.diffusions[0]

    def reconcile_targets(self, *images):
        """ 
        This function is responsible for approximating any primes needed then creating their approx derived images, and returning those derived images 
        This default, null-illusion implementation simply returns the input images with no changes
        """
        return list(images)

    @property
    def num_derived_images(self):
        return len(inspect.signature(self.reconcile_targets).parameters)

    def encode_images_in_parallel(self, images):
        assert len(images) == self.num_derived_images
        def encode(diffusion, image):
            return diffusion.encode_image(image)
        return rp.par_map(encode, self.diffusions, images, num_threads=self.num_threads)

    def decode_latents_in_parallel(self, latents):
        assert len(latents) == self.num_derived_images
        def decode(diffusion, latent):
            return diffusion.decode_latent(latent)
        return rp.par_map(decode, self.diffusions, latents, num_threads=self.num_threads)


    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None, decode=True):
        assert len(prompts) == self.num_derived_images, f'len(prompts)={len(prompts)}  !=  num_derived_images={self.num_derived_images}'

        text_embeddings = [diffusion.get_text_embedding(prompt) for diffusion, prompt in zip(self.diffusions, prompts)] #This is vary fast

        #UNCOMMENT TO USE DIFFERENT INITIAL NOISES
        # latents = latents if latents is not None else [
        #     torch.randn(4, 64, 64) for x in prompts
        # ]

        #UNCOMMENT TO USE SAME INITIAL NOISES
        latents = latents if latents is not None else [
            torch.randn(4, 64, 64)
        ] * len(prompts)

        #Set the devices properly
        latents = [
            latent.to(diffusion.device, torch.float32)
            for latent, diffusion in zip(latents, self.diffusions)
        ]


        for diffusion in self.diffusions:
            diffusion.scheduler.set_timesteps(num_steps)

        # # Add some timesteps to the beginning...
        # timesteps_list = as_numpy_array(self.scheduler.timesteps).tolist()
        # self.scheduler.timesteps = torch.tensor(
        #     sorted(sorted(timesteps_list)[-5:] * 5 + timesteps_list, reverse=True),
        #     dtype=self.scheduler.timesteps.dtype,
        #     device=self.scheduler.timesteps.device,
        # )
        # fansi_print(f"TIMESTEPS: {self.scheduler.timesteps}", "green bold")

        for t in tqdm(self.primary_diffusion.timesteps):

            def pred(latent, text_embedding, diffusion):
                noise_pred = diffusion.pred_noise(latent, t.to(diffusion.device), guidance_scale, text_embedding)

                clean_pred = get_clean_sample(
                    sample         = latent,
                    model_output   = noise_pred,
                    pred_type      = "epsilon",
                    timestep       = t.to(diffusion.device),
                    alphas_cumprod = diffusion.alphas_cumprod,
                )

                image_pred = diffusion.decode_latent(clean_pred)
                image_pred = image_pred.clamp(0,1)

                return noise_pred, clean_pred, image_pred

            preds = rp.par_map(pred, latents, text_embeddings, self.diffusions, num_threads=self.num_threads)

            noise_preds, clean_preds, image_preds = zip(*preds)

            derived_images = self.reconcile_targets(*image_preds)
            assert all(old.device==new.device for old, new in zip(image_preds, derived_images)), 'self.reconcile_targets should NOT change the devices of the images!'

            derived_clean_preds = self.encode_images_in_parallel(derived_images)

            #TRY: Shuffle noise components...what happens?
            #noise_preds = [x.to(y.device) for x,y in zip(shuffled(noise_preds),noise_preds)] #For hidden overlays, shuffle noises...

            #TRY: Average noise componenents...I think it's slightly worse than shuffling, BUT it still looks better than nothing?
            # noise_preds = [x.to(y.device) for x,y in zip([mean(x.to(self.primary_diffusion.device) for x in noise_preds)]*len(noise_preds),noise_preds)] #For hidden overlays - merge noises by averaging

            noise_preds = [
                get_epsilon(
                    sample=latent,
                    model_output=derived_clean_pred,
                    pred_type="sample",
                    timestep=t,
                    alphas_cumprod=diffusion.alphas_cumprod,
                )
                for latent, derived_clean_pred, diffusion in zip(latents, derived_clean_preds, self.diffusions)
            ]

            for i, (latent, noise_pred, diffusion) in enumerate(zip(latents, noise_preds, self.diffusions)):
                latent = diffusion.scheduler.step(noise_pred, t, latent).prev_sample
                latents[i]=latent

        if not decode:
            return latents

        all_images = self.decode_latents_in_parallel(latents)

        #Convert from CHW torch tensors to HWC numpy arrays
        all_images = rp.as_numpy_images(all_images)

        return all_images

def _reconcile_hidden_overlays_initial(Ta, Tb, Tc, Td, Tz, Lz, backlight):
    """
    Closed-form initial estimate of A, B, C, D, Z minimizing weighted mean squared error.

    Parameters
    ----------
    Ta, Tb, Tc, Td : float
        Target values for A, B, C, D.
    Tz : float
        Target value for Z.
    Lz : float
        Weight for how much to prioritize Z reconstruction.
    backlight : float
        Multiplicative constant in the definition of Z = backlight * A * B * C * D.

    Returns
    -------
    list of float
        [A, B, C, D, Z] initial estimates.
    """
    p = Ta * Tb * Tc * Td
    epsilon = Tz - backlight * p
    v = [1 / Ta, 1 / Tb, 1 / Tc, 1 / Td]
    v_dot_v = sum(vi**2 for vi in v)
    scaling = (backlight * Lz * p * epsilon) / (
        1 + (backlight**2) * Lz * p**2 * v_dot_v
    )
    delta = [scaling * vi for vi in v]
    A = Ta + delta[0]
    B = Tb + delta[1]
    C = Tc + delta[2]
    D = Td + delta[3]
    Z = backlight * A * B * C * D
    return [A, B, C, D, Z]


def reconcile_hidden_overlays(Ta, Tb, Tc, Td, Tz, Lz=3, backlight=3):
    """
    Refined estimate of A, B, C, D, Z by gradient steps starting from closed-form initialization.
    Math done with mathematica + chatGPT: https://chatgpt.com/share/680fbe46-239c-8006-89c7-87f32a381c5c

    Note: With a higher backlight value, you can get better accuracy for free!
    HOWEVER: There's a tradeoff: Higher backlight values don't model real-world overlays as well, as innacuracies in the printing process are exacerbated a lot
        A backlight value of 3 is the most you'd really want to use...backlight value of 2 is safe for real-world use

    NOTE: Lz is the Loss-coefficient for image Z. Basically, if it's higher - we prioritize the accuracy of Z more than the accuracy of A,B,C,D
    If Lz=0, then it returns exactly A=Ta,B=Tb,C=Tc,D=Td - not useful, resulting in no change.

    Hidden overlay illusion:
    Given 5 target images, we want to solve for prime images A,B,C,D
    We define derived image Z = A * B * C * D * backlight (where backlight is the brightness of the light behind the overlays)
    This solves for A,B,C,D using least squares, such that:
        |  GIVEN:
        |  Ta Tb Tc Td Tz, Lz (Lz is a coefficient for how much we relatively care about the Z reconstruction)
        |  (Here, Ta for example is an atomic variable - it's not like T * a, it's just T_a shorthand)
        |
        |  RELATIONSHIPS TO A,B,C,D,Z:
        |  A = Ta
        |  B = Tb
        |  C = Tc
        |  D = Td
        |  Z = 3 * A * B * C * D
        |
        |  GOAL:
        |  Solve for A, B, C, D, Z
        |  Minimize Mean Squared Error:
        |  (Ta - A)^2 +
        |  (Tb - B)^2 +
        |  (Tc - C)^2 +
        |  (Td - D)^2 +
        |  Lz * (Tz - Z)^2

    Parameters
    ----------
    Ta, Tb, Tc, Td : float
        Target values for A, B, C, D.
    Tz : float
        Target value for Z.
    Lz : float, optional
        Weight for Z reconstruction (default is 1).
    backlight : float, optional
        Multiplicative constant in Z = backlight * A * B * C * D (default is 3).

    Returns
    -------
    list of float
        [A, B, C, D, Z] refined estimates.
    """

    #We use an initial estimate to speed it up. We could start from just A=Ta, B=Tb, C=Tc, D=Td but
    #   that's slower than using a good first guess, provided with the below function
    A, B, C, D, _ = _reconcile_hidden_overlays_initial(Ta, Tb, Tc, Td, Tz, Lz, backlight=backlight)

    max_iter = 30
    step_size = .01
    for _ in range(max_iter):
        Z = backlight * A * B * C * D
        err = Tz - Z

        loss_grad_A = -2 * (Ta - A) + (-2 * backlight * Lz * err * B * C * D)
        loss_grad_B = -2 * (Tb - B) + (-2 * backlight * Lz * err * A * C * D)
        loss_grad_C = -2 * (Tc - C) + (-2 * backlight * Lz * err * A * B * D)
        loss_grad_D = -2 * (Td - D) + (-2 * backlight * Lz * err * A * B * C)

        A -= step_size * loss_grad_A
        B -= step_size * loss_grad_B
        C -= step_size * loss_grad_C
        D -= step_size * loss_grad_D

        #Just make sure there's no NaN pixels, or pixels outside the range [0,1]
        A = rp.r._nan_to_num(rp.r._clip(A,0,1))
        B = rp.r._nan_to_num(rp.r._clip(B,0,1))
        C = rp.r._nan_to_num(rp.r._clip(C,0,1))
        D = rp.r._nan_to_num(rp.r._clip(D,0,1))


    Z = backlight * A * B * C * D
    return [A, B, C, D, Z]

def demo_reconcile_hidden_overlays():
    #Below we demo the above functions, displaying the target image on the left and the best-fit overlays on the right
    images = [
        "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/bernese-mountain-dog.jpg?crop=1.00xw:0.668xh;0,0.252xh&resize=640:*",
        "https://www.princeton.edu/sites/default/files/styles/1x_full_2x_half_crop/public/images/2022/02/KOA_Nassau_2697x1517.jpg?itok=Bg2K7j7J",
        "https://money.com/wp-content/uploads/2024/03/Best-Small-Dog-Breeds-Pomeranian.jpg?quality=60",
        "https://money.com/wp-content/uploads/2024/03/Best-Small-Dog-Breeds-Maltese.jpg?quality=60",
        "https://www.dogstrust.org.uk/images/800x600/assets/2025-03/toffee%202.jpg",
    ]
    images = rp.load_images(images, use_cache=True)
    images = rp.crop_images_to_square(images)
    images = rp.resize_images_to_min_size(images)
    images = rp.as_float_images(images)
    images = rp.as_rgb_images(images)
    Ta, Tb, Tc, Td, Tz = images
    A, B, C, D, Z = rp.gather_args_call(reconcile_hidden_overlays)

    rp.display_image(
        rp.grid_concatenated_images(
            rp.list_transpose(
                [
                    rp.labeled_images([Ta, Tb, Tc, Td, Tz], ["Ta", "Tb", "Tc", "Td", "Tz"]),
                    rp.labeled_images([A, B, C, D, Z], ["A", "B", "C", "D", "Z"]),
                ]
            )
        ),
        block=False,
    )

class FlipIllusion(DiffusionIllusion):

    def reconcile_targets(self, image_a, image_b):
        """ This function is responsible for approximating any primes needed then creating their approx derived images, and returning those derived images """
        def flip_image(image):
            return image.flip(1,2)

        merged_image = (image_a + flip_image(image_b).to(image_a.device)) / 2

        new_image_a = merged_image
        new_image_b = flip_image(merged_image).to(image_b.device)

        if rp.toc()>5:
            rp.display_image(
                rp.tiled_images(
                    rp.as_numpy_images(
                        [new_image_a, new_image_b],
                    ),
                    length=2,
                )
            )
            rp.tic()

        return [new_image_a, new_image_b]


class HiddenOverlayIllusion(DiffusionIllusion):

    def reconcile_targets(self, image_a, image_b, image_c, image_d, image_z):
        """ This function is responsible for approximating any primes needed then creating their approx derived images, and returning those derived images """

        new_image_a, new_image_b, new_image_c, new_image_d, new_image_z = reconcile_hidden_overlays(
            image_a.to(image_a.device),
            image_b.to(image_a.device),
            image_c.to(image_a.device),
            image_d.to(image_a.device),
            image_z.to(image_a.device),
        )

        new_image_a = new_image_a.to(image_a.device)
        new_image_b = new_image_b.to(image_b.device)
        new_image_c = new_image_c.to(image_c.device)
        new_image_d = new_image_d.to(image_d.device)
        new_image_z = new_image_z.to(image_z.device)

        output = [new_image_a, new_image_b, new_image_c, new_image_d, new_image_z]

        if rp.toc()>5:
            rp.display_image(
                rp.tiled_images(
                    rp.as_numpy_images(
                        [image_a, image_b, image_c, image_d, image_z] + [*output]
                    ),
                    length=len(output),
                )
            )
            rp.tic()

        return output


if __name__ == "__main__":
    if not 'illusion_pairs' in vars():
        illusion_pairs = []

    #FOR ITERM2 REMOTE ACCESS
    #rp.display_image = rp.display_image_in_terminal_imgcat

    ##SELECT DEVICES BASED ON YOUR SYSTEM
    # devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:0'] ; parallel=True  #RLab GPU's
    devices = ['mps'   , 'mps'   , 'mps'   , 'mps'   , 'mps'   ] ; parallel=False #Macbook

    # #REGULAR DIFFUSION
    # illusion = DiffusionIllusion(
    #     [
    #         Diffusion(device=devices[0]),
    #     ],
    #     parallel=parallel,
    # )

    #FLIPPY ILLUSIONS
    illusion = FlipIllusion(
        [
            Diffusion(device=devices[0]),
            Diffusion(device=devices[1]),
        ],
        parallel=parallel,
    )

    # #HIDDEN OVERLAY ILLUSIONS
    # illusion = HiddenOverlayIllusion(
    #     [
    #         Diffusion(device=devices[0]),
    #         Diffusion(device=devices[1]),
    #         Diffusion(device=devices[2]),
    #         Diffusion(device=devices[3]),
    #         Diffusion(device=devices[4]),
    #     ],
    #     parallel=parallel,
    # )
    
    for _ in range(100):
        with torch.no_grad():

            prompts = [
                "Oil painting of a Chicken",
                "professional portrait photograph of a gorgeous Norwegian girl in winter clothing with long wavy blonde hair, freckles, gorgeous symmetrical face, cute natural makeup, wearing elegant warm winter fashion clothing, ((standing outside))",
                # "A orange cute kitten in a cardboard box in times square",
                # "Walter white, oil painting, octane render, 8 0 s camera, portrait",
                "Oil painting of a cat",
                "Oil painting of Golden Retriever",
                "Hatsune miku, gorgeous, amazing, elegant, intricate, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by ross tran",
                # "Hatsune miku, gorgeous, amazing, elegant, intricate, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by ross tran",
                # " mario 3d nintendo video game",
                # "Hatsune miku, gorgeous, amazing, elegant, intricate, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by ross tran",
                # "Hatsune miku, gorgeous, amazing, elegant, intricate, highly detailed, digital painting, artstation, concept art, sharp focus, illustration, art by ross tran",
                # "Still of jean - luc picard in star trek = the next generation ( 1 9 8 7 )",
                # "An intricate HB pencil sketch of a giraffe head",
                # "An intricate HB pencil sketch of a penguin",
                # "Pixel art sprite of a Golden Retriever",
                # "mario"
            ]

            prompts = rp.random_batch(prompts, illusion.num_derived_images)
            rp.fansi_print(
                f'PROMPTS:\n{rp.indentify(rp.line_join(prompts),"    - ")}', "cyan gray"
            )


            #TODO: Maybe mix a bit of pure noise back in again?
            num_steps=20
            num_repeats=4
            renoise_alpha=0
            
            for repeat in range(num_repeats):
            
                latents = illusion.sample(
                    prompts,
                    guidance_scale=10,
                    latents = None if repeat==0 else latents,
                    num_steps=num_steps,
                    decode=False,
                )

                """
                TODO:
                    Some questions:
                        is the CFG compounded after each inversion? by how much, perhaps measure repeated CFG/inversions variance and plot against variance of different CFG's?
                            If we use lower CFG's at the beginning repeats, maybe we can make more elegant merges? cause its super biased towards making cat with particular eyes in particular place for example...maybe better diffusion model would handle that better tho...
                        EDICT vs DDIM inversion...is EDICT better for this case? Does it matter? (cause it IS slower...)
                        Can we start inversion from maybe the middle step instead of going all the way to the end? 
                        Can we do 5, 10, 20, then 40 diffusion steps (then would be O(n) computation, 2n to be specific)
                             could do that via first 1/4, first 1/2, then 1 of steps...
                        Does tiny renoise alpha really destroy it that badly? (seems so...) what if we add renoise alpha to later diffusion steps an not earlier ones, where the tweedies are able to sync better (adding just to the noise component)? (would have to do it ONCE at some step because DDIM...)\
                        What if we increase CFG at every repeat?
                        What if we do inversion properly, i.e. with the prompts and CFG and everything it supports?
                """
                if repeat < num_repeats-1:
                    latents = [
                        # illusion.diffusions[0].edict_inversion(latent, num_steps)
                        illusion.diffusions[0].ddim_inversion(latent, num_steps)
                        for latent in rp.eta(latents,'Inverting')
                    ]  # Should done in parallel TODO
                    

                    def blend_noise(noise_background, noise_foreground, alpha):
                        """ Variance-preserving blend """
                        return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5

                    def mix_new_noise(noise, alpha):
                        """As alpha --> 1, noise is destroyed"""
                        if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise)      , alpha)
                        elif isinstance(noise, np.ndarray): return blend_noise(noise, np.random.randn(*noise.shape), alpha)
                        else: raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")
                    latents = [mix_new_noise(latent, renoise_alpha) for latent in latents]

    
            images = illusion.decode_latents_in_parallel(latents)
            images = rp.as_numpy_images(images)

            illusion_pairs.append(images)
            rp.display_image(rp.horizontally_concatenated_images(images))
            image_paths = rp.save_images(
                list(images)
                + [rp.horizontally_concatenated_images(rp.as_numpy_images(images))]
            )
            rp.fansi_print(
                f'SAVED IMAGES:\n{rp.indentify(rp.line_join(image_paths), "    • ")}',
                "green bold",
            )
