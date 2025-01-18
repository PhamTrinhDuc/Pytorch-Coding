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
        self.stride = stride # for getattr 
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding=padding)
        self.residual = VAEResidualBlock(in_channels=out_channels, output_channels=out_channels)

    def forward(self, x):
        # [B, in_channels, H, W] => [B, out_channels, H, W] 
        x = self.conv1(x)
        # [B, out_channels, H, W] => # [B, out_channels, H, W]
        x = self.residual(x)
        return x

class VaeEncoder(nn.Sequential):
    def __init__(self,
                 in_encode: int, 
                 hidden_encode: int=128, 
                 out_encode: int=4,
                 num_groups: int = 32):
        self.out_encode = out_encode
        super().__init__(
            # [B, 3, H, W] => [B, hidden_encode, H, W] 
            BlockEncoder(in_channels=in_encode, 
                         out_channels=hidden_encode, 
                         kernel_size=3, 
                         stride=1, 
                         padding=1),
            # [B, hidden_encode, H, W] => [B, hidden_encode*2, H//2, W//2]
            BlockEncoder(in_channels=hidden_encode, 
                         out_channels=hidden_encode*2, 
                         kernel_size=3, 
                         stride=2, 
                         padding=0),
            # [B, hidden_encode*2, H//2, W//2] => [B, hidden_encode*4, H//4, W//4]
            BlockEncoder(in_channels=hidden_encode*2, 
                         out_channels=hidden_encode*4, 
                         kernel_size=3, 
                         stride=2, 
                         padding=0),
            # [B, hidden_encode*4, H//4, W//4] => [B, hidden_encode*4, H//8, W//8]
            BlockEncoder(in_channels=hidden_encode*4, 
                         out_channels=hidden_encode*4, 
                         kernel_size=3, 
                         stride=2, 
                         padding=0),
            # [B, hidden_encode*4, H//8, W//8] => [B, hidden_encode*4, H//8, W//8]
            VAEResidualBlock(in_channels=hidden_encode*4, 
                             output_channels=hidden_encode*4, 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            # [B, hidden_encode*4, H//8, W//8] => [B, hidden_encode*4, H//8, W//8]
            VAEAttentionBlock(channels=hidden_encode*4),
            # [B, hidden_encode*4, H//8, W//8] => [B, hidden_encode*4, H//8, W//8]
            VAEResidualBlock(in_channels=hidden_encode*4, 
                             output_channels=hidden_encode*4, 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            # [B, hidden_encode*4, H//8, W//8] => [B, hidden_encode*4, H//8, W//8]
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_encode*4),
            # [B, hidden_encode*4, H//8, W//8] => [B, hidden_encode*4, H//8, W//8]
            nn.SiLU(),
            # [B, hidden_encode*4, H//8, W//8] => [B, out_encode*2, H//8, W//8]
            nn.Conv2d(in_channels=hidden_encode*4, 
                      out_channels=out_encode*2, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            # [B, out_encode*2, H//8, W//8] => [B, out_encode*2, H//8, W//8]
            nn.Conv2d(in_channels=out_encode*2, 
                      out_channels=out_encode*2, 
                      kernel_size=1, 
                      padding=0, 
                      stride=1)
        )
    
    def forward(self, x: torch.Tensor):
        # x: (B, in_channels, H, W)
        # noise: (B, out_encode//2, H//8, W//8)
        B, C, H, W = x.size()

        for module in self:
            if getattr(module, "stride", None) == 2:
                # The image shape after passing through the module will be reduced to [(H/2)-1, (W/2)-1], 
                # so before passing through the module, we will add 1 to H and W in the lower right corner 
                # to ensure say after going through the module the image shape is [H/2, W/2]
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # [B, out_encode*2, H//8, W//8] => 2 tensors [B, out_encode, H//8, W//8]
        mean, log_variance = torch.chunk(input=x, chunks=2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # [B, out_encode, H//8, W//8] => [B, out_encode, H//8, W//8]
        log_variance = torch.clamp(log_variance, -30, 20)
        # (B, out_encode, H//8, W//8) -> (B, out_encode, H//8, W//8)
        variance = log_variance.exp()
        # (B, out_encode, H//8, W//8) -> (B, out_encode, H//8, W//8)
        stdev = variance.sqrt()

        # Transform N(0, 1) -> N(mean, stdev) 
        noise = torch.randn(size=(B, self.out_encode, H//8, W//8))
        # (B, out_encode, H//8, W//8) -> (B, out_encode, H//8, W//8)
        x = mean + stdev * noise
        # Scale by a constant
        return x * 0.18125, mean, log_variance
    

def main():
    batch_size = 1
    in_channels = 3
    hidden_channels = 128
    out_channels = 4
    W_image = H_image = 224
    num_groups = 32
    data = torch.randn((batch_size, in_channels, W_image, H_image))
    encoder = VaeEncoder(in_encode=in_channels,
                         hidden_encode=hidden_channels, 
                         out_encode=out_channels, 
                         num_groups=num_groups)
    output, *_ = encoder(data)
    print(output.shape)

if __name__ == "__main__":
    main()