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
        self.timesteps = torch.from_numpy(ndarray=np.arange(0, num_training_steps)[::-1].copy())

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

    def add_noise(self):
        pass
