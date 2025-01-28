import torch
import numpy as np

class DDPMSampler:
    def __init__(self, 
                 generator: torch.Generator,
                 num_training_steps: int=1000,
                 beta_start: float=0.00085, 
                 beta_end: float=0.0120):
        # [num_traning_steps, ]
        self.beta = torch.linspace(start=beta_start**0.5, 
                                   end=beta_end**0.5, 
                                   steps=num_training_steps, 
                                   dtype=torch.float32)**0.5
        self.alphas = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(input=self.alphas, dim=0)
        self.ones = torch.tensor([1.0])

        self.num_traning_steps = num_training_steps
        self.generator = generator
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_time(self, ):
        pass

    def set_strength(self):
        pass

    def _get_previous_timesteps(self, timestep: int) -> int:
        pass

    def _get_variance(self, timesteps: int) -> torch.Tensor:
        pass

    def step(self):
        pass


    def add_noise(self, 
                  original_samples: torch.FloatTensor,
                  timestep: torch.IntTensor) -> torch.FloatTensor:
        """forward process"""

        # tensor([1])
        sqrt_alpha_prod = self.alpha_cumprod[timestep] ** 0.5
        # tensor([1])
        # sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # [1] -> [1, 1, 1, 1]
        # while(len(sqrt_alpha_prod) < len(original_samples)):
        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        # tensor([1])
        sqrt_one_minus_alpha_prod = (1 - self.alpha_cumprod[timestep]) ** 0.5
        # tensor([1])
        # sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # [1] -> [1, 1, 1, 1]
        # while(len(sqrt_one_minus_alpha_prod) < len(original_samples)):
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # from images/kernel_trick.png
        noise = torch.randn(size=original_samples.shape,
                            generator=self.generator,
                            device=original_samples.device,
                            dtype=original_samples.dtype)
        # [B, 4, latent_H, latent_W] -> [B, 4, latent_H, latent_W
        noise_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noise_samples # [B, 4, latent_H, latent_W
    

def debug():
    sampler = DDPMSampler(
        generator=torch.Generator(device="cpu")
    )
    latents = torch.randn(size=(1, 4, 64, 64))
    noise_added = sampler.add_noise(original_samples=latents,
                                    timestep=torch.tensor([1]))
    print(noise_added.shape)

if __name__ == "__main__":
    debug()