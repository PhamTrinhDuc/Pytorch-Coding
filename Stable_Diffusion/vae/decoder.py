import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from template import VAEAttentionBlock, VAEResidualBlock


class DecoderBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int,
                 padding: int, 
                 is_up_scale: bool = True):
        super().__init__()
        self.is_up_scale = is_up_scale
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.residual = VAEResidualBlock(in_channels=out_channels, 
                                         output_channels=out_channels)
        self.up_sample = nn.Upsample(scale_factor=2)
    
    def forward(self, x: torch.Tensor):
        # [B, C, H, W] => [B, C, H, W]
        x = self.conv(x)
        # [B, C, H, W] => [B, C, H, W]
        for _ in range(3):
            x = self.residual(x)
        if self.is_up_scale:
            # [B, C, H, W] => [B, C, H*2, W*2]
            x = self.up_sample(x)
        return x # [B, C, H*2, W*2]


class VaeDecoder(nn.Module):
    def __init__(self, 
                 out_decode: int,
                 in_decode: int=4, 
                 hidden_decode: int=512,
                 num_groups: int=32):
        super().__init__()

        self.layers = nn.ModuleList([
            # [B, in_decode, H//8, W//8] => [B, in_decode, H//8, W//8]
            nn.Conv2d(in_channels=in_decode, 
                      out_channels=in_decode, 
                      kernel_size=1, 
                      padding=0),
            # [B, in_decode, H//8, W//8] => [B, hidden_decode, H//8, W//8]
            nn.Conv2d(in_channels=in_decode, 
                      out_channels=hidden_decode, 
                      kernel_size=3, 
                      padding=1),
            # [B, hidden_decode, H//8, W//8] => [B, hidden_decode, H//8, W//8]
            VAEResidualBlock(in_channels=hidden_decode, 
                             output_channels=hidden_decode),
            # [B, hidden_decode, H//8, W//8] => [B, hidden_decode, H//8, W//8]
            VAEAttentionBlock(channels=hidden_decode),
            # [B, hidden_decode, H//8, W//8] => [B, hidden_decode, H//8, W//8]
            *[VAEResidualBlock(in_channels=hidden_decode, 
                               output_channels=hidden_decode) for _ in range(4)],
            nn.Upsample(scale_factor=2),
            # [B, hidden_decode, H//4, W//4] => [B, hidden_decode, H//2, W//2]
            DecoderBlock(in_channels=hidden_decode, 
                         out_channels=hidden_decode, 
                         kernel_size=3, 
                         stride=1, 
                         padding=1), 

            # [B, hidden_decode, H//2, W//2] => [B, hidden_decode/2, H, W]
            DecoderBlock(in_channels=hidden_decode, 
                         out_channels=hidden_decode//2, 
                         kernel_size=3,
                         stride=1, 
                         padding=1),

            # [B, hidden_decode/2, H, W] => [B, hidden_decode/4, H, W]
            DecoderBlock(in_channels=hidden_decode//2, 
                         out_channels=hidden_decode//4, 
                         kernel_size=3, 
                         stride=1, 
                         padding=1,
                         is_up_scale=False),
            # [B, hidden_decode/4, H, W] => [B, hidden_decode/4, H, W]
            nn.GroupNorm(num_groups=num_groups, 
                         num_channels=hidden_decode//4),

            # [B, hidden_decode/4, H, W] => [B, hidden_decode/4, H, W]
            nn.SiLU(),

            # [B, hidden_decode/4, H, W] => [B, out_decode, H, W]
            nn.Conv2d(in_channels=hidden_decode//4,
                      out_channels=out_decode,
                      kernel_size=3,
                      padding=1)

        ])

    def forward(self, x: torch.Tensor):
        x /= 0.18125
        # [B, in_decode, H//8, W//8] => [B, out_decode, H, W]
        for module in self.layers:
            x = module(x)
        return x # [B, out_decode, H, W]
    
def main():
    batch_size = 1
    in_decode = 4
    hidden_decode = 512
    out_decode = 3
    num_groups = 32
    W_image = H_image = 224
    decoder = VaeDecoder(in_decode=in_decode, 
                         out_decode=out_decode, 
                         hidden_decode=hidden_decode, 
                         num_groups=num_groups)
    mock_data = torch.randn(size=(batch_size, in_decode, W_image//8, H_image//8))
    output = decoder(mock_data)
    print(output.shape)


if __name__ == "__main__":
    main()