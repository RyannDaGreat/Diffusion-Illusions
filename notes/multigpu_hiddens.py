import inspect
from functools import cached_property

import torch
import torch.nn as nn
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

import numpy as np

import rp

#PLEASE REPLACE YOUR DIFFUSERS DDIM SCHEDULER WITH THIS ONE
# class DDIMScheduler(SchedulerMixin, ConfigMixin):
#     """
#     `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
#     non-Markovian guidance.
#
#     This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
#     methods the library implements for all schedulers such as loading and saving.
#
#     Args:
#         num_train_timesteps (`int`, defaults to 1000):
#             The number of diffusion steps to train the model.
#         beta_start (`float`, defaults to 0.0001):
#             The starting `beta` value of inference.
#         beta_end (`float`, defaults to 0.02):
#             The final `beta` value.
#         beta_schedule (`str`, defaults to `"linear"`):
#             The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
#             `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
#         trained_betas (`np.ndarray`, *optional*):
#             Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
#         clip_sample (`bool`, defaults to `True`):
#             Clip the predicted sample for numerical stability.
#         clip_sample_range (`float`, defaults to 1.0):
#             The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
#         set_alpha_to_one (`bool`, defaults to `True`):
#             Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
#             there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
#             otherwise it uses the alpha value at step 0.
#         steps_offset (`int`, defaults to 0):
#             An offset added to the inference steps, as required by some model families.
#         prediction_type (`str`, defaults to `epsilon`, *optional*):
#             Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
#             `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
#             Video](https://imagen.research.google/video/paper.pdf) paper).
#         thresholding (`bool`, defaults to `False`):
#             Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
#             as Stable Diffusion.
#         dynamic_thresholding_ratio (`float`, defaults to 0.995):
#             The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
#         sample_max_value (`float`, defaults to 1.0):
#             The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
#         timestep_spacing (`str`, defaults to `"leading"`):
#             The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
#             Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
#         rescale_betas_zero_snr (`bool`, defaults to `False`):
#             Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
#             dark samples instead of limiting it to samples with medium brightness. Loosely related to
#             [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
#     """
#
#     _compatibles = [e.name for e in KarrasDiffusionSchedulers]
#     order = 1
#
#     @register_to_config
#     def __init__(
#         self,
#         num_train_timesteps: int = 1000,
#         beta_start: float = 0.0001,
#         beta_end: float = 0.02,
#         beta_schedule: str = "linear",
#         trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
#         clip_sample: bool = True,
#         set_alpha_to_one: bool = True,
#         steps_offset: int = 0,
#         prediction_type: str = "epsilon",
#         thresholding: bool = False,
#         dynamic_thresholding_ratio: float = 0.995,
#         clip_sample_range: float = 1.0,
#         sample_max_value: float = 1.0,
#         timestep_spacing: str = "leading",
#         rescale_betas_zero_snr: bool = False,
#     ):
#         if trained_betas is not None:
#             self.betas = torch.tensor(trained_betas, dtype=torch.float32)
#         elif beta_schedule == "linear":
#             self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
#         elif beta_schedule == "scaled_linear":
#             # this schedule is very specific to the latent diffusion model.
#             self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
#         elif beta_schedule == "squaredcos_cap_v2":
#             # Glide cosine schedule
#             self.betas = betas_for_alpha_bar(num_train_timesteps)
#         else:
#             raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
#
#         # Rescale for zero SNR
#         if rescale_betas_zero_snr:
#             self.betas = rescale_zero_terminal_snr(self.betas)
#
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#
#         # At every step in ddim, we are looking into the previous alphas_cumprod
#         # For the final step, there is no previous alphas_cumprod because we are already at 0
#         # `set_alpha_to_one` decides whether we set this parameter simply to one or
#         # whether we use the final alpha of the "non-previous" one.
#         self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
#
#         # standard deviation of the initial noise distribution
#         self.init_noise_sigma = 1.0
#
#         # setable values
#         self.num_inference_steps = None
#         self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
#
#     def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
#         """
#         Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
#         current timestep.
#
#         Args:
#             sample (`torch.FloatTensor`):
#                 The input sample.
#             timestep (`int`, *optional*):
#                 The current timestep in the diffusion chain.
#
#         Returns:
#             `torch.FloatTensor`:
#                 A scaled input sample.
#         """
#         return sample
#
#     def _get_variance(self, timestep, prev_timestep):
#         alpha_prod_t = self.alphas_cumprod[timestep]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
#         beta_prod_t = 1 - alpha_prod_t
#         beta_prod_t_prev = 1 - alpha_prod_t_prev
#
#         variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
#
#         return variance
#
#     # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
#     def _threshold_sample(self, sample: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
#         prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
#         s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
#         pixels from saturation at each step. We find that dynamic thresholding results in significantly better
#         photorealism as well as better image-text alignment, especially when using very large guidance weights."
#
#         https://arxiv.org/abs/2205.11487
#         """
#         dtype = sample.dtype
#         batch_size, channels, *remaining_dims = sample.shape
#
#         if dtype not in (torch.float32, torch.float64):
#             sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half
#
#         # Flatten sample for doing quantile calculation along each image
#         sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
#
#         abs_sample = sample.abs()  # "a certain percentile absolute pixel value"
#
#         s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
#         s = torch.clamp(
#             s, min=1, max=self.config.sample_max_value
#         )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
#         s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
#         sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"
#
#         sample = sample.reshape(batch_size, channels, *remaining_dims)
#         sample = sample.to(dtype)
#
#         return sample
#
#     def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
#         """
#         Sets the discrete timesteps used for the diffusion chain (to be run before inference).
#
#         Args:
#             num_inference_steps (`int`):
#                 The number of diffusion steps used when generating samples with a pre-trained model.
#         """
#
#         if num_inference_steps > self.config.num_train_timesteps:
#             raise ValueError(
#                 f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
#                 f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
#                 f" maximal {self.config.num_train_timesteps} timesteps."
#             )
#
#         self.num_inference_steps = num_inference_steps
#
#         # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
#         if self.config.timestep_spacing == "linspace":
#             timesteps = (
#                 np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
#                 .round()[::-1]
#                 .copy()
#                 .astype(np.int64)
#             )
#         elif self.config.timestep_spacing == "leading":
#             step_ratio = self.config.num_train_timesteps // self.num_inference_steps
#             # creates integer timesteps by multiplying by ratio
#             # casting to int to avoid issues when num_inference_step is power of 3
#             timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
#             timesteps += self.config.steps_offset
#         elif self.config.timestep_spacing == "trailing":
#             step_ratio = self.config.num_train_timesteps / self.num_inference_steps
#             # creates integer timesteps by multiplying by ratio
#             # casting to int to avoid issues when num_inference_step is power of 3
#             timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
#             timesteps -= 1
#         else:
#             raise ValueError(
#                 f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
#             )
#
#         self.timesteps = torch.from_numpy(timesteps).to(device)
#
#     def step(
#         self,
#         model_output: torch.FloatTensor,
#         timestep: int,
#         sample: torch.FloatTensor,
#         eta: float = 0.0,
#         use_clipped_model_output: bool = False,
#         generator=None,
#         variance_noise: Optional[torch.FloatTensor] = None,
#         return_dict: bool = True,
#         prev_timestep = None,
#     ) -> Union[DDIMSchedulerOutput, Tuple]:
#         """
#         Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
#         process from the learned model outputs (most often the predicted noise).
#
#         Args:
#             model_output (`torch.FloatTensor`):
#                 The direct output from learned diffusion model.
#             timestep (`float`):
#                 The current discrete timestep in the diffusion chain.
#             sample (`torch.FloatTensor`):
#                 A current instance of a sample created by the diffusion process.
#             eta (`float`):
#                 The weight of noise for added noise in diffusion step.
#             use_clipped_model_output (`bool`, defaults to `False`):
#                 If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
#                 because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
#                 clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
#                 `use_clipped_model_output` has no effect.
#             generator (`torch.Generator`, *optional*):
#                 A random number generator.
#             variance_noise (`torch.FloatTensor`):
#                 Alternative to generating noise with `generator` by directly providing the noise for the variance
#                 itself. Useful for methods such as [`CycleDiffusion`].
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.
#
#         Returns:
#             [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
#                 If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
#                 tuple is returned where the first element is the sample tensor.
#
#         """
#         if self.num_inference_steps is None:
#             raise ValueError(
#                 "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
#             )
#
#         # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
#         # Ideally, read DDIM paper in-detail understanding
#
#         # Notation (<variable name> -> <name in paper>
#         # - pred_noise_t -> e_theta(x_t, t)
#         # - pred_original_sample -> f_theta(x_t, t) or x_0
#         # - std_dev_t -> sigma_t
#         # - eta -> η
#         # - pred_sample_direction -> "direction pointing to x_t"
#         # - pred_prev_sample -> "x_t-1"
#
#         # 1. get previous step value (=t-1)
#         if prev_timestep is None:
#             prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
#
#         print("STEPS: ",timestep, prev_timestep)
#
#         # 2. compute alphas, betas
#         alpha_prod_t = self.alphas_cumprod[timestep]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
#
#         beta_prod_t = 1 - alpha_prod_t
#
#         # 3. compute predicted original sample from predicted noise also called
#         # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#         if self.config.prediction_type == "epsilon":
#             pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
#             pred_epsilon = model_output
#         elif self.config.prediction_type == "sample":
#             pred_original_sample = model_output
#             pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
#         elif self.config.prediction_type == "v_prediction":
#             pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
#             pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
#         else:
#             raise ValueError(
#                 f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
#                 " `v_prediction`"
#             )
#
#         # 4. Clip or threshold "predicted x_0"
#         if self.config.thresholding:
#             pred_original_sample = self._threshold_sample(pred_original_sample)
#         elif self.config.clip_sample:
#             pred_original_sample = pred_original_sample.clamp(
#                 -self.config.clip_sample_range, self.config.clip_sample_range
#             )
#
#         # 5. compute variance: "sigma_t(η)" -> see formula (16)
#         # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
#         variance = self._get_variance(timestep, prev_timestep)
#         sign = torch.sign(variance)
#         variance = torch.abs(variance)
#
#         std_dev_t = eta * variance ** (0.5) * sign
#
#         if use_clipped_model_output:
#             # the pred_epsilon is always re-derived from the clipped x_0 in Glide
#             pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
#
#         # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#         pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
#
#         # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
#         prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
#
#         if eta > 0:
#             assert False
#             # if variance_noise is not None and generator is not None:
#             #     raise ValueError(
#             #         "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
#             #         " `variance_noise` stays `None`."
#             #     )
#             #
#             # if variance_noise is None:
#             #     variance_noise = randn_tensor(
#             #         model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
#             #     )
#             # variance = std_dev_t * variance_noise
#             #
#             # prev_sample = prev_sample + variance
#
#         if not return_dict:
#             return (prev_sample,)
#
#         return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
#
#     # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
#     def add_noise(
#         self,
#         original_samples: torch.FloatTensor,
#         noise: torch.FloatTensor,
#         timesteps: torch.IntTensor,
#     ) -> torch.FloatTensor:
#         # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
#         # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
#         # for the subsequent add_noise calls
#         self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
#         alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
#         timesteps = timesteps.to(original_samples.device)
#
#         sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
#         sqrt_alpha_prod = sqrt_alpha_prod.flatten()
#         while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
#             sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
#
#         sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
#         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
#         while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
#             sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
#
#         noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
#         return noisy_samples
#
#     # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity
#     def get_velocity(
#         self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
#     ) -> torch.FloatTensor:
#         # Make sure alphas_cumprod and timestep have same device and dtype as sample
#         self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
#         alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
#         timesteps = timesteps.to(sample.device)
#
#         sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
#         sqrt_alpha_prod = sqrt_alpha_prod.flatten()
#         while len(sqrt_alpha_prod.shape) < len(sample.shape):
#             sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
#
#         sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
#         sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
#         while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
#             sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
#
#         velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
#         return velocity
#
#     def __len__(self):
#         return self.config.num_train_timesteps
#
#
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
    def ddim_inversion(self, image, num_steps: int = 50) -> torch.Tensor:
        # This function is NOT thread safe if used concurrently on the same GPU due to the TemporarilySetAttr
        latents = self.encode_image(image)[None]

        with rp.TemporarilySetAttr(self.pipe, scheduler=self.inverse_scheduler):
            inv_latents, _ = self.pipe(
                prompt="",
                negative_prompt="",
                guidance_scale=1.0,
                width=rp.get_image_width(image),
                height=rp.get_image_height(image),
                output_type="latent",
                return_dict=False,
                num_inference_steps=num_steps,
                latents=latents,
            )

        inv_latent = inv_latents[0]
        return inv_latent

    @torch.no_grad()
    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None):
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
    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None, do_x:bool=True, do_y:bool=True):
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

    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None):
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


def index_linterp(length, scheme):
    """
    Generate zigzag indices based on a discrete scheme.
    
    Args:
        length: Target length of output array
        scheme: List defining zigzag pattern
                [0, 1] for regular (0,1,2,3,...)
                [1, 0] for backwards (length-1, length-2, ...)
                [0, 1, 0] for forward then backwards
    
    Returns:
        Array of zigzag indices
    """
    if not scheme or length <= 0:
        return np.array([])
    
    if len(scheme) == 1:
        return np.array([int(scheme[0] * (length - 1))] * length)
    
    result = []
    
    for i in range(length):
        # Normalize position across the entire scheme (0 to 1)
        if length == 1:
            t_global = 0.0
        else:
            t_global = i / (length - 1)
        
        # Map to scheme position (0 to len(scheme) - 1)
        scheme_pos = t_global * (len(scheme) - 1)
        
        # Find which segment we're in
        seg_idx = int(scheme_pos)
        if seg_idx >= len(scheme) - 1:
            seg_idx = len(scheme) - 2
        
        # Local position within the segment (0 to 1)
        t_local = scheme_pos - seg_idx
        
        # Interpolate between scheme values
        start_val = scheme[seg_idx]
        end_val = scheme[seg_idx + 1]
        scheme_value = start_val + t_local * (end_val - start_val)
        
        # Convert to actual index
        result.append(int(scheme_value * (length - 1)))
    
    return np.array(result)

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


    def sample(self, prompts: list[str], num_steps=20, guidance_scale=7.5, latents=None):
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

            assert len(diffusion.scheduler.timesteps) == num_steps
            # diffusion.scheduler.timesteps = rp.resize_list(rp.resize_list(diffusion.scheduler.timesteps, num_steps//4),num_steps)

            N=7
            diffusion.scheduler.timesteps = torch.cat(
                tuple(
                    x.flip(0) if i!=N-1 else x
                    for i,x in enumerate(rp.split_into_n_sublists(diffusion.scheduler.timesteps, N))
                )
            )



            # zigzag=[0,1] #Default
            # zigzag=[0,.5,0,1] #Halfway Zigzag
            # # zigzag = [
            # #     0 / 4,
            # #     1 / 4,
            # #     0 / 4,
            # #     2 / 4,
            # #     1 / 4,
            # #     3 / 4,
            # #     2 / 4,
            # #     4 / 4,
            # # ]  # Halfway Zigzag
            # # zigzag=[1,0,1] #Weirdness - its bad
            # # zigzag=[0,1,0,1] #Complete Zigzag
            # diffusion.scheduler.timesteps = diffusion.scheduler.timesteps[index_linterp(num_steps,zigzag)]

            assert len(diffusion.scheduler.timesteps) == num_steps

            print(diffusion.scheduler.timesteps)

        # # Add some timesteps to the beginning...
        # timesteps_list = as_numpy_array(self.scheduler.timesteps).tolist()
        # self.scheduler.timesteps = torch.tensor(
        #     sorted(sorted(timesteps_list)[-5:] * 5 + timesteps_list, reverse=True),
        #     dtype=self.scheduler.timesteps.dtype,
        #     device=self.scheduler.timesteps.device,
        # )
        # fansi_print(f"TIMESTEPS: {self.scheduler.timesteps}", "green bold")

        for timestep_index in tqdm(range(num_steps)):
            print("----")

            #I need it to be more intelligent with timesteps
            timestep = self.primary_diffusion.scheduler.timesteps[timestep_index]
            if timestep_index < len(self.primary_diffusion.scheduler.timesteps) -1:
                prev_timestep = self.primary_diffusion.scheduler.timesteps[timestep_index+1]
            else:
                prev_timestep = 0
            print('PRED PREV',prev_timestep)

            def pred(latent, text_embedding, diffusion):

                noise_pred = diffusion.pred_noise(latent, timestep.to(diffusion.device), guidance_scale, text_embedding)

                clean_pred = get_clean_sample(
                    sample         = latent,
                    model_output   = noise_pred,
                    pred_type      = "epsilon",
                    timestep       = timestep.to(diffusion.device),
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
                    timestep=timestep,
                    alphas_cumprod=diffusion.alphas_cumprod,
                )
                for latent, derived_clean_pred, diffusion in zip(latents, derived_clean_preds, self.diffusions)
            ]

            for i, (latent, noise_pred, diffusion) in enumerate(zip(latents, noise_preds, self.diffusions)):
                latent = diffusion.scheduler.step(noise_pred, timestep, latent, prev_timestep=prev_timestep).prev_sample
                latents[i]=latent

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
    rp.seed_all(42)    

    if not 'illusion_pairs' in vars():
        illusion_pairs = []

    #FOR ITERM2 REMOTE ACCESS
    rp.display_image = rp.display_image_in_terminal_imgcat

    ##SELECT DEVICES BASED ON YOUR SYSTEM
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:0'] ; parallel=True  #RLab GPU's
    #devices = ['mps'   , 'mps'   , 'mps'   , 'mps'   , 'mps'   ] ; parallel=False #Macbook

    #REGULAR DIFFUSION
    illusion = DiffusionIllusion(
        [
            Diffusion(device=devices[0]),
        ]
    )

    # #FLIPPY ILLUSIONS
    # illusion = FlipIllusion(
    #     [
    #         Diffusion(device=devices[0]),
    #         Diffusion(device=devices[1]),
    #     ]
    # )

    #HIDDEN OVERLAY ILLUSIONS
    illusion = HiddenOverlayIllusion(
        [
            Diffusion(device=devices[0]),
            Diffusion(device=devices[1]),
            Diffusion(device=devices[2]),
            Diffusion(device=devices[3]),
            Diffusion(device=devices[4]),
        ],
        parallel=parallel,
    )

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

            images = illusion.sample(
                prompts,
                guidance_scale=10,
                num_steps=20,
                # num_steps=40,
                # num_steps=80,
            )

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


