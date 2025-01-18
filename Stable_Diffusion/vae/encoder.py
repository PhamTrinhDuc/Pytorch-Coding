import torch
import torch.nn as nn
import torch.nn.functional as F
from template import VAEResidualBlock, VAEAttentionBlock


class BlockEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.residual = VAEResidualBlock(in_channels=in_channels, output_channels=out_channels)

    def forward(self, x):
        # [B, in_channels, H, W] => [B, out_channels, H, W] 
        x = self.conv1(x)
        # [B, out_channels, H, W] => # [B, out_channels, H, W]
        x = self.residual(x)
        return x

class VaeEncoder(nn.Sequential):
    def __init__(self, in_channels: int=3, out_channels: int=128, num_groups: int=32):
        super().__init__(
            # [B, 3, H, W] => [B, 128, H, W] 
            BlockEncoder(in_channels=in_channels, 
                         out_channels=out_channels, 
                         kernel_size=3, 
                         stride=1, 
                         padding=1),
            # [B, 128, H, W] => [B, 128, H//2, W//2]
            BlockEncoder(in_channels=out_channels, 
                         out_channels=out_channels*2, 
                         kernel_size=3, 
                         stride=2, 
                         padding=0),
            # [B, 256, H//2, W//2] => [B, 512, H//4, W//4]
            BlockEncoder(in_channels=out_channels*2, 
                         out_channels=out_channels*3, 
                         kernel_size=3, 
                         stride=2, 
                         padding=0),
            # [B, 512, H//4, W//4] => [B, 512, H//8, W//8]
            BlockEncoder(in_channels=out_channels*3, 
                         out_channels=out_channels*3, 
                         kernel_size=3, 
                         stride=2, 
                         padding=0),
            # [B, 512, H//8, W//8] => [B, 512, H//8, W//8]
            VAEResidualBlock(in_channels=out_channels*3, 
                             output_channels=out_channels*3, 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            # [B, 512, H//8, W//8] => [B, 512, H//8, W//8]
            VAEAttentionBlock(channels=out_channels*3),
            # [B, 512, H//8, W//8] => [B, 512, H//8, W//8]
            VAEResidualBlock(in_channels=out_channels*3, 
                             output_channels=out_channels*3, 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            # [B, 512, H//8, W//8] => [B, 512, H//8, W//8]
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels*3),
            nn.SiLU(),
            # [B, 512, H//8, W//8] => [B, 8, H//8, W//8]
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1),
            # [B, 8, H//8, W//8] => [B, 8, H//8, W//8]
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0, stride=1)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        # x: (B, C, H, W)
        # noise: (B, 4, H / 8, W / 8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # The image shape after passing through the module will be reduced to [(H/2)-1, (W/2)-1], 
                # so before passing through the module, we will add 1 to H and W in the lower right corner 
                # to ensure say after going through the module the image shape is [H/2, W/2]
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # [B, 8, H//8, W//8] => 2 tensors [B, 4, H//8, W//8]
        mean, log_variance = torch.chunk(input=x, chunks=2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # [B, 4, H//8, W//8] => [B, 4, H//8, W//8]
        log_variance = torch.clamp(log_variance, -30, 20)
        # (B, 4, H//8, W//8) -> (B, 4, H//8, W//8)
        variance = log_variance.exp()
        # (B, 4, H//8, W//8) -> (B, 4, H//8, W//8)
        stdev = variance.sqrt()

        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        # Scale by a constant
        return x * 0.18125
    

def main():
    data = torch.randn((1, 3, 224, 224))
    noise = torch.randn((1, 8, 224//8, 224//8))
    encoder = VaeEncoder(in_channels=3, out_channels=128)
    output = encoder(data, noise)
    print(output.shape)

if __name__ == "__main__":
    main()