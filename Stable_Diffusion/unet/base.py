import torch
import torch.nn.functional as F
from torch import nn
from unet.template import UnetAttentionBlock, UnetResidualBlock


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


class DownSample(nn.Module):
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
    

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # x: [B, 4, H/8, W/8]: output of ther VAE encoder
        # context: [B, seq_len_KV, dim_KV]
        # time: [B, 1, 1280]

        for layer in self:
            if isinstance(layer, UnetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UnetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
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
            SwitchSequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=channels, 
                          kernel_size=3, 
                          padding=1)
            ),
            # 1. [B, channels, H/8, W/8] => [B, channels, H/8, W/8]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels, 
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model, 
                                   num_heads=num_heads),
            ),
            # [B, channels, H/8, W/8] => [B, channels, H/16, W/16]
            SwitchSequential(
                DownSample(channels=channels),
            ),
            # 2. [B, channels, H/16, W/16] => [B, channels*2, H/16, W/16]
            # because channels = d_model * num_heads so when channels increase 2 to then d_model increase 2
            SwitchSequential(
                UnetResidualBlock(in_channels=channels, 
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
            ),
            # [B, channels, H/16, W/16] => [B, channels*2, H/16, W/16]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*2, 
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
            ),
            # [B, channels*2, H/16, W/16] => [B, channels*2, H/32, W/32]
            SwitchSequential(
                DownSample(channels=channels*2),
            ),
            # 3. [B, channels*2, H/32, W/32] => [B, channels*4, H/32, W/32]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*2, 
                                  out_channels=channels*4),
                UnetAttentionBlock(d_model=d_model*4, 
                                   num_heads=num_heads),
            ),
            # [B, channels*4, H/32, W/32] => [B, channels*4, H/32, W/32]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*4, 
                                  out_channels=channels*4),
                UnetAttentionBlock(d_model=d_model*4, 
                                   num_heads=num_heads),
            ),
            # [B, channels*4, H/32, W/32] => [B, channels*4, H/64, W/64]
            SwitchSequential(
                DownSample(channels=channels*4),
            ),
            # 4. pass through 2 Residuals block 
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            SwitchSequential(
                *[UnetResidualBlock(in_channels=channels*4, 
                                    out_channels=channels*4)
                                    for _ in range(2)]
            ),
        ])

        self.bottle_neck = nn.ModuleList([
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            *[SwitchSequential(
                UnetResidualBlock(in_channels=channels*4, 
                                  out_channels=channels*4),
                UnetAttentionBlock(d_model=d_model*4, 
                                   num_heads=num_heads),
            ) for _ in range(2)],
        ])

        # self.decoders = nn.ModuleList([
        #     # 4'. [B, channel*4*2, H/64, W/64] => [B, channels*4, H/64, W/64]
        #     # becauce inpute decoder concat the skip connection of the encoder 
        #     *[UnetResidualBlock(in_channels=channels*4*2,
        #                       out_channels=channels*4)],
        #     Upsample(channels=channels*4*2),
        #     UnetBlock(in_channels=channels*4*2,
        #               out_channels=channels*4,
        #               d_model=d_model*4,
        #               num_heads=num_heads),
        #     # [B, channels*4, H/64, W/64] => [B, channels*4, H/32, W/32]
        #     Upsample(channels=channels*4),

        #     # 3'. [B, channels*4, H/32, W/32] => [B, channels*2, H/16, W/16]
        #     *[UnetBlock(in_channels=channels*4, 
        #                 out_channels=channels*2)],
        # ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # x: [B, 4, H/8, W/8]: output of ther VAE encoder
        # context: [B, seq_len_KV, dim_KV]
        # time: [B, 1, 1280]

        skip_connection =[]
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connection.append(x)
        
        for i, x in enumerate(skip_connection):
            print(f"{i+1}: {x.shape}")


def main():
    batch_size = 1
    d_model = 320
    d_context = 768
    seq_len = 77
    num_heads = 1
    n_times = 1280
    num_groups = 32
    H_img = W_img = 512
    unet = Unet(d_model=d_model, num_heads=num_heads)

    x = torch.randn(size=(batch_size, 4, H_img//8, W_img//8))
    time = torch.randn(size=(batch_size, 1, n_times))
    context = torch.randn(size=(batch_size, seq_len, d_context))
    output = unet(x, context, time)

if __name__ == "__main__":
    main()