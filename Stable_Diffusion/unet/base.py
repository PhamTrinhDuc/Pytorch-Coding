import torch
import torch.nn.functional as F
from torch import nn
from template import UnetAttentionBlock, UnetResidualBlock


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        self.linear1 = nn.Linear(in_features=d_model, 
                                 out_features=d_model*4)
        self.linear2 = nn.Linear(in_features=d_model*4, 
                                 out_features=d_model)
        self.silu = nn.SiLU()
        
    def forward(self, x: torch.Tensor):
        # shape x: [1, d_model]
        # [1, d_model] => [1, d_model*4]
        x = self.linear1(x)
        # [1, d_model] => [1, d_model*4]
        x = self.silu(x)
        # [1, d_model*4] => [1, d_model]
        x = self.linear2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, 
                                        kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
        )
            
    def forward(self, x: torch.Tensor):
        # [B, channels, H, W] => [B, channels, H*2, W*2]
        x = self.blocks(x)
        return x        


class DownSamle(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels=channels, 
                              out_channels=channels, 
                              kernel_size=3,
                              stride=2,
                              padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor):
        # [B, channels, H, W] => [B, channels, H*2, W*2]
        x = self.blocks(x)
        return x
    

class UnetBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 d_model: int,
                 num_heads: int,
                 num_groups: int=32,
                 n_times: int=1280):
        super().__init__()

        self.residual = UnetResidualBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          num_groups=num_groups,
                                          n_times=n_times)
        self.attention = UnetAttentionBlock(d_model=d_model,
                                            num_heads=num_heads,
                                            num_groups=num_groups)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # [B, in_channels, H, W] => [B, out_channels, H, W]
        x = self.residual(x, time)
        # out_channels = num_heads * d_model
        # [B, out_channels, H, W] => [B, out_channels, H, W]
        x = self.attention(x, context)
        return x
        

class Unet(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 in_channels: int=4,
                 num_groups: int=32, 
                 n_times: int=1280):
        super().__init__()
        channels = num_heads * d_model
        self.encoders = nn.ModuleList([

            # [B, in_channels, H/8, W/8] => [B, channels, H/8, W/8]
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=channels, 
                      kernel_size=3, 
                      padding=1),

            # 1. [B, channels, H/8, W/8] => [B, channels, H/8, W/8]
            *[UnetBlock(in_channels=channels, 
                        out_channels=channels,
                        d_model=d_model, 
                        num_heads=num_heads) for _ in range(2)],
            # [B, channels, H/8, W/8] => [B, channels, H/16, W/16]
            DownSamle(channels=channels),

            # 2. [B, channels, H/16, W/16] => [B, channels*2, H/16, W/16]
            # because channels = d_model * num_heads so when channels increase 2 to then d_model increase 2
            UnetBlock(in_channels=channels, 
                      out_channels=channels*2, 
                      d_model=d_model*2, 
                      num_heads=num_heads),
            UnetBlock(in_channels=channels*2,
                      out_channels=channels*2,
                      d_model=d_model*2, 
                      num_heads=num_heads),
            # [B, channels*2, H/16, W/16] => [B, channels*2, H/32, W/32]
            DownSamle(channels=channels*2),

            # 3. [B, channels*2, H/32, W/32] => [B, channels*4, H/32, W/32]
            UnetBlock(in_channels=channels*2,
                      out_channels=channels*4, 
                      d_model=d_model*4, 
                      num_heads=num_heads),
            # [B, channels*4, H/32, W/32] => [B, channels*4, H/32, W/32]
            UnetBlock(in_channels=channels*4,
                      out_channels=channels*4,
                      d_model=d_model*4,
                      num_heads=num_heads),
            # [B, channels*4, H/32, W/32] => [B, channels*4, H/64, W/64]
            DownSamle(channels=channels*4),

            # 4. pass through 2 Residuals block 
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            *[UnetResidualBlock(in_channels=channels*4, 
                                out_channels=channels*4)
                                for _ in range(2)]
        ]),

        self.bottle_neck = nn.ModuleList([
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            UnetBlock(in_channels=channels*4,
                      out_channels=channels*4,
                      d_model=d_model*4,
                      num_heads=num_heads),
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            UnetBlock(in_channels=channels*4,
                      out_channels=channels*4,
                      num_heads=num_heads,
                      num_heads=num_heads)
        ])

        self.decoders = nn.ModuleList([
            # 4'. [B, channel*4*2, H/64, W/64] => [B, channels*4, H/64, W/64]
            # becauce inpute decoder concat the skip connection of the encoder 
            UnetBlock(in_channels=channels*4*2,
                      out_channels=channels*4,
                      d_model=d_model*4,
                      num_heads=num_heads),
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/32, W/32]
            Upsample(channels=channels*4),

            # 3'. [B, channels*4, H/32, W/32] => [B, channels*2, H/16, W/16]
            UnetBlock(in_channels=channels*4, )
        ])

    def forward(self, x):
        pass



def main():
    d_model = 768
    n_times = 1280
    num_groups = 32
    time_embed = TimeEmbedding(d_model=d_model)
    unet_residual = UnetResidualBlock(in_channels=32, 
                                      out_channels=64, 
                                      num_groups=num_groups,
                                      n_times=n_times)

if __name__ == "__main__":
    main()