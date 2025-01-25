import torch
import torch.nn as nn
from config import UnetArgs
from unet.base import TimeEmbedding, Unet, UnetOutLayer


class Diffusion(nn.Module):
    def __init__(self, config: UnetArgs):
        super().__init__()
        self.time_embed = TimeEmbedding(n_times=config.n_times)

        self.unet = Unet(d_model=config.d_model, 
                         num_heads=config.num_heads,
                         in_channels=config.hidden_channels, 
                         num_groups=config.num_groups,
                         n_times=config.n_times*4)
        
        self.out_layer = UnetOutLayer(in_channels=config.num_heads * config.d_model,
                                      out_channels=config.hidden_channels, 
                                      num_groups=config.num_groups)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, times: torch.Tensor): 
        # latent: (output of the encoder VAE). Shape: [B, config.hidden_channels, H/8, W/8]
        # context: (output of the CLIP, usefull for cross-attention). Shape: [B, seq_len_KV, dim_KV]
        # time: [B, 1, config.n_times]

        
        # [B, 1, config.n_times] => [B, 1, config.n_times*4]
        time_embed = self.time_embed(times)
        print("time embedding: ", time_embed.shape)
        # [B, config.hidden_channels, H/8, W/8] => [B, d_model*num_heads, H/8, W/8]
        output = self.unet(latent, context, time_embed)
        print("output unet: ", output.shape)
        # [B, d_model*num_heads, H/8, W/8] => [B, config.hidden_channels, H/8, W/8]
        # output = self.out_layer(output)
        return output



def main():
    batch_size = 1
    d_model = 320
    d_context = 768
    seq_len = 77
    num_heads = 1
    n_times = 320
    num_groups = 32
    H_img = W_img = 512
    config = UnetArgs()
    unet = Diffusion(config).to(config.device)

    x = torch.randn(size=(batch_size, 4, H_img//8, W_img//8)).to(config.device)
    time = torch.randn(size=(batch_size, 1, n_times)).to(config.device)
    context = torch.randn(size=(batch_size, seq_len, d_context)).to(config.device)
    output = unet(x, context, time)
    print(output.shape)


if __name__ == "__main__":
    main()