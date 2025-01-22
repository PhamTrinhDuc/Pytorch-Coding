from dataclasses import dataclass

@dataclass
class UnetArgs: 
    d_model = 160
    num_heads = 2
    hidden_channels = 4 # output channels of encoder VAE 
    num_groups = 32
    n_times = 320

@dataclass
class VaeArgs:
    in_channels = 3
    hidden_channels = 128
    out_channels = 4 # in_channels of the Unet's input
    num_groups = 32