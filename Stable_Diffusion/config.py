from dataclasses import dataclass
import torch


@dataclass
class UnetArgs: 
    d_model = 320
    num_heads = 1
    hidden_channels = 4 # output channels of encoder VAE 
    num_groups = 32
    n_times = 320
    device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VaeArgs:
    in_channels = 3 # channels of image
    hidden_encode_channels = 128
    hidden_decode_channels = 512
    out_channels = 4 # in_channels of the Unet's input
    num_groups = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CLIPArgs:
    vocab_size = 49408
    d_model = 768
    seq_len = 77
    num_heads = 1
    num_layers = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DiffusionArgs:
    IMAGE_HEIGHT = IMAGE_WIDTH = 512
    LATENT_HEIGHT = LATENT_WIDTH = IMAGE_HEIGHT // 8
    strength: float=0.8,
    do_cfg: bool=True,
    cfg_scale: float=7.5,
    sampler_name: str="ddpm",
    n_inference_steps: int=50,
    models: dict={},
    seed=None,
    device: str="cpu",
    idle_device: str="cpu",
    tokenier=None