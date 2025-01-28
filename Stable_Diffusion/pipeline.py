import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from ddpm import DDPMSampler
from config import DiffusionArgs


args = DiffusionArgs()


def rescale(x, old_range: tuple, new_range: tuple, clamp: bool=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def process_prompt_input(
        do_cfg: bool, 
        cond_prompt: str, 
        uncond_prompt: str, 
        tokenizer,
        clip_model
    ) -> torch.FloatTensor:

    context = None
    if do_cfg:
        # [B, seq_len = 77]
        prompt_tokens = tokenizer.batch_encode_plus(cond_prompt,
                                                    padding="max_length", 
                                                    max_length=77).input_ids
        # [B, seq_len=77]
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
        # [B, seq_len=77, d_model]
        prompt_context = clip_model(prompt_tokens)
        
        # [B, seq_len=77]
        uncond_prompt_tokens =tokenizer.batch_encode_plus(uncond_prompt,
                                                          padding="max_length", 
                                                          max_length=77).input_ids
        # [B, seq_len=77]
        uncond_prompt_tokens = torch.tensor(uncond_prompt_tokens, dtype=torch.long)
        # [B, seq_len=77, d_model]
        uncond_prompt_context = clip_model(uncond_prompt_tokens)
        
        context = torch.cat([prompt_context, uncond_prompt_context])
    
    else:
        # [B, seq_len = 77]
        prompt_tokens = tokenizer.batch_encode_plus(cond_prompt,
                                                    padding="max_length", 
                                                    max_length=77).input_ids
        # [B, seq_len=77]
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
        # [B, seq_len=77, d_model]
        context = clip_model(prompt_tokens)

    return context


def process_image_input(input_image, 
                        sampler, 
                        encoder,
                        generator, 
                        device, 
                        strength):
    latent_shape = (1, 4, args.LATENT_HEIGHT, args.LATENT_WIDTH)
    if input_image:
        # [H, W, C]
        input_image_tensor = input_image.resize(args.IMAGE_HEIGHT, args.IMAGE_WIDTH)
        # [H, W, C]
        input_image_tensor = np.array(input_image_tensor)
        # [H, W, C]
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
        # [H, W, C]
        input_image_tensor = rescale(x=input_image_tensor, old_range=(0, 255), new_range=(-1, 1))
        # [H, W, C] -> [B, H, W, C] -> [B, C, H, W]
        input_image_tensor = input_image_tensor.unqueeze(0).permute(0, 3, 1, 2)
        # [B, 4, latent_H, latent_W]
        latents = encoder(input_image_tensor)
        # add noise to the latents (the encoder input image)
        sampler.set_strength(strength=strength)
        # [B, 4, latent_H, latent_W] -> [B, 4, latent_H, latent_W]
        latents = sampler.add_noise(latents, sampler.timesteps[0])

    else:
        # [B, 4, latent_H, latent_W]
        latents = torch.randn(size=(latent_shape), generator=generator, device=device) 
    
    return latents


def generate(prompt: str,
             uncond_prompt: str = None,
             input_image=None,
             strength: float=0.8,
             do_cfg: bool=True,
             cfg_scale: float=7.5,
             sampler_name: str="ddpm",
             n_inference_steps: int=50,
             models: dict={},
             seed=None,
             device: str="cpu",
             idle_device: str="cpu",
             tokenizer=None):
    """

    Generate image from arguments 

    Args:
        prompt (str): The input text describes what you want the model to generate.
        uncond_prompt (str): Used to guide the model in the absence of specific constraints.
        input_image: An original image used as input to create new content.
        strength (float): Scale adjusts the influence of the original image (if using input_image).
                0.0: Completely retains the original image.
                1.0: Completely ignores the original image, relying only on noise to create a new image.
        do_cfg (classifier-free guidance): Indicates whether Classifier-Free Guidance (CFG) is used or not. 
                If true, The model will combine conditional (according to prompt) and unconditional (according to uncond_prompt) results to create a picture.
        cfg_scale (float): Level of influence of Classifier-Free Guidance (when do_cfg=True).
                For example: With cfg_scale=7.5, the model will generate more detailed images based on the prompt.
        sampler_name (str): The name of the sampler used to generate the data.
        n_inference_steps (int): Number of steps in the reversed process.
        model: A set of loaded models (may contain diffusion model checkpoints).
        seed (int): The same prompt and seed will produce the same result.
        idle_device (str): The backup device is used when the primary device is busy.

    """

    with torch.no_grad():
        if not 0 < strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        # func for backup device 
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.seed(seed)
        else:
            generator.manual_seed(seed=seed)
        
        # init sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timestep(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        # init model for reverse process
        clip = models['clip'].to(device)
        encoder = models['encoder'].to(device)
        diffusion = models['diffusion'].to(device)
        decoder = models['decoder'].to(device)

        # get context from cond_prompt and uncond_prompt
        context = process_prompt_input(do_cfg=do_cfg, 
                                       cond_prompt=prompt, 
                                       uncond_prompt=uncond_prompt, 
                                       tokenizer=tokenizer, 
                                       clip_model=clip)
        to_idle(clip)
        # get latents from input image or None with encoder VAE
        latents = process_image_input(input_image=input_image, 
                                      sampler=sampler,
                                      encoder=encoder, 
                                      generator=generator,
                                      device=device,
                                      strength=strength)
        to_idle(encoder)

        # implement reverse process
        timesteps = tqdm(sampler.timesteps, desc="time steps for reverse process")
        
        for i, step in enumerate(timesteps):
            
            # [1, 320]
            time_embedding = get_time_embedding(step)

            if do_cfg: # if have prompt and uncond_prompt
                # [B, 4, latent_H, latent_W]
                input_model = latents
                # [B, 4, latent_H, latent_W] => [B*2, 4, latent_H, latent_W]
                #I do this because the output will return for uncondional and conditonal
                input_model = input_model.repeat(2, 1, 1, 1)

            output_model = diffusion(input_model, context, time_embedding)

            if do_cfg: # if have prompt and uncond_prompt
                out_cond, out_uncond = output_model.chunk(2)
                model_output_cfg = cfg_scale * (out_cond - out_uncond) + out_uncond
            
            latents = sampler.step(step, latents, model_output_cfg)
        to_idle(diffusion)

        # decode latents to the image output
        images = decoder(latents)
        images = rescale(x=images, old_range=(-1, 1), new_range=(0, 255), clamp=True)
        # [B, C, H, W] = > [B, H, W, C]
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]