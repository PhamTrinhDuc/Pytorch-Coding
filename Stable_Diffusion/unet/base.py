import torch
import torch.nn.functional as F
from torch import nn
from unet.template import UnetAttentionBlock, UnetResidualBlock
from config import UnetArgs

class TimeEmbedding(nn.Module):
    def __init__(self, n_times: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=n_times, 
                                 out_features=n_times*4)
        self.linear2 = nn.Linear(in_features=n_times*4, 
                                 out_features=4*n_times)
        self.silu = nn.SiLU()
        
    def forward(self, x: torch.Tensor):
        # shape x: [1, n_times]
        # [1, n_times] => [1, n_times*4]
        x = self.linear1(x)
        # [1, n_times] => [1, n_times*4]
        x = self.silu(x)
        # [1, n_times*4] => [1, n_times*4]
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
        # time: [B, 1, n_times*4]

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
                 in_channels: int=4, # output channels of the VAE encoder
                 num_groups: int=32, 
                 n_times: int=320*4):
        super().__init__()
        channels = num_heads * d_model

        self.encoders = nn.ModuleList([

            # 1. [B, in_channels, H/8, W/8] => [B, channels, H/8, W/8]
            SwitchSequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=channels, 
                          kernel_size=3, 
                          padding=1)
            ),
            # 2. [B, channels, H/8, W/8] => [B, channels, H/8, W/8]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels, 
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model, 
                                   num_heads=num_heads),
            ),
            # 3. [B, channels, H/8, W/8] => [B, channels, H/8, W/8]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels, 
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model, 
                                   num_heads=num_heads),
            ),
            # 4. [B, channels, H/8, W/8] => [B, channels, H/16, W/16]
            SwitchSequential(
                DownSample(channels=channels),
            ),
            # 5. [B, channels, H/16, W/16] => [B, channels*2, H/16, W/16]
            # because channels = d_model * num_heads so when channels increase 2 to then d_model increase 2
            SwitchSequential(
                UnetResidualBlock(in_channels=channels, 
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
            ),
            # 6. [B, channels*2, H/16, W/16] => [B, channels*2, H/16, W/16]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*2, 
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
            ),
            # 7. [B, channels*2, H/16, W/16] => [B, channels*2, H/32, W/32]
            SwitchSequential(
                DownSample(channels=channels*2),
            ),
            # 8. [B, channels*2, H/32, W/32] => [B, channels*4, H/32, W/32]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*2, 
                                  out_channels=channels*4),
                UnetAttentionBlock(d_model=d_model*4, 
                                   num_heads=num_heads),
            ),
            # 9. [B, channels*4, H/32, W/32] => [B, channels*4, H/32, W/32]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*4, 
                                  out_channels=channels*4),
                UnetAttentionBlock(d_model=d_model*4, 
                                   num_heads=num_heads),
            ),
            # 10. [B, channels*4, H/32, W/32] => [B, channels*4, H/64, W/64]
            SwitchSequential(
                DownSample(channels=channels*4),
            ),
            # 11. pass through 2 Residuals block 
            # [B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            SwitchSequential(UnetResidualBlock(in_channels=channels*4, 
                                               out_channels=channels*4)
            ),
            # 12.[B, channels*4, H/64, W/64] => [B, channels*4, H/64, W/64]
            SwitchSequential(UnetResidualBlock(in_channels=channels*4, 
                                               out_channels=channels*4)
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

        """
        Output of encoder layers: 
            1: torch.Size([1, 320, 32, 32])
            2: torch.Size([1, 320, 32, 32])
            3: torch.Size([1, 320, 32, 32])
            4: torch.Size([1, 320, 16, 16])
            5: torch.Size([1, 640, 16, 16])
            6: torch.Size([1, 640, 16, 16])
            7: torch.Size([1, 640, 8, 8])
            8: torch.Size([1, 1280, 8, 8])
            9: torch.Size([1, 1280, 8, 8])
            10: torch.Size([1, 1280, 4, 4])
            11: torch.Size([1, 1280, 4, 4])
            12: torch.Size([1, 1280, 4, 4])
        """
        self.decoders = nn.ModuleList([
        
            # 12'. [B, channel*4*2, H/64, W/64] => [B, channels*4, H/64, W/64]
            # becauce inpute decoder concat the skip connection of the encoder 
            SwitchSequential(UnetResidualBlock(in_channels=channels*4*2,
                                               out_channels=channels*4) 
            ),
            # 11'. [B, channel*4*2, H/64, W/64] => [B, channels*4, H/64, W/64]
             SwitchSequential(UnetResidualBlock(in_channels=channels*4*2,
                                               out_channels=channels*4),
            ),
            # 10'. [B, channels*4*2, H/64, W/64] => [B, channels*4, H/32, W/32]
             SwitchSequential(
                UnetResidualBlock(in_channels=channels*4*2,
                                  out_channels=channels*4),
                Upsample(channels=channels*4)
            ),
            # 9'. [B, channels*4*2, H/32, W/32] => [B, channels*4, H/32, W/32]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*4*2, 
                                  out_channels=channels*4),
                UnetAttentionBlock(d_model=d_model*4,
                                   num_heads=num_heads),
            ),
            # 8'. [B, channels*4*2, H/32, W/32] => [B, channels*4, H/32, W/32]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*4*2, 
                                  out_channels=channels*4), 
                UnetAttentionBlock(d_model=d_model*4, 
                                   num_heads=num_heads),
            ),
            # 7'. [B, channels*6, H/32, W/32] => [B, channels*4, H/32, W/32] => [B, channels*4, H/16, W/16]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*6,
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
                Upsample(channels=channels*2),
            ),
            # 6'. [B, channels*6, H/16, W/16] => [B, channels*4, H/16, W/16]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*4,
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
            ),
            # 5'. [B, channels*6, H/16, W/16] => [B, channels*2, H/16, W/16]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*4,
                                  out_channels=channels*2),
                UnetAttentionBlock(d_model=d_model*2, 
                                   num_heads=num_heads),
            ),
            # 4'. # [B, channel*3, H/16, W/16] => [B, channels, H/16, W/16] => [B, channels, H/8, W/8]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*3,
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model,
                                   num_heads=num_heads),
                Upsample(channels=channels),
            ), 
            # 3'. [B, channels*2, H/8, W/8] => [B, channels, H/8, W/8]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*2,
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model,
                                   num_heads=num_heads), 
            ),
            # 2'. [B, channels*2, H/8, W/8] => [B, channels, H/8, W/8]
            SwitchSequential(
                UnetResidualBlock(in_channels=channels*2,
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model,
                                   num_heads=num_heads)
            ),
            # 1'. [B, channels*2, H/8, W/8] => [B, channels, H/8, W/8]
             SwitchSequential(
                UnetResidualBlock(in_channels=channels*2,
                                  out_channels=channels),
                UnetAttentionBlock(d_model=d_model,
                                   num_heads=num_heads)
            )
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # x: [B, 4, H/8, W/8]: output of ther VAE encoder
        # context: [B, seq_len_KV, dim_KV]
        # time: [B, 1, 1280]

        skip_connection =[]
        # print("Output of encoder layers: ")
        for i, layer in enumerate(self.encoders):
            x = layer(x, context, time)
            # print(f"{i+1}: {x.shape}")
            skip_connection.append(x)

        # print("\nOutput of the bottle neck: ")
        for i, layer in enumerate(self.bottle_neck):
            x = layer(x, context, time)
            # print(f"{i+1}: {x.shape}")

        # print("\nOutput of the encoder layers: ")
        for i, layer in enumerate(self.decoders):
            x = torch.cat([x, skip_connection.pop()], dim=1)
            x = layer(x, context, time)
            # print(f"{i+1}': {x.shape}")
        return x


class UnetOutLayer(nn.Module):
    def __init__(self, 
                 in_channels: int=320, 
                 out_channels: int=4, 
                 num_groups: int=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv_out = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=out_channels,
                                  kernel_size=3, 
                                  padding=1)
    
    def forward(self, x: torch.Tensor):
        # x: output of Unet, shape: [B, in_channels, H/8, W/8]
        # [B, in_channels, H/8, W/8] => [B, in_channels, H/8, W/8]
        x = self.norm(x)
        # [B, in_channels, H/8, W/8] => [B, out_channels, H/8, W/8]
        x = self.conv_out(x)
        return x


def main():
    batch_size = 1
    d_model = 320
    batch_size = 1
    d_model = 320
    d_context = 768
    seq_len = 77
    num_heads = 1
    n_times = 1280
    num_groups = 32
    H_img = W_img = 256
    unet = Unet(d_model=d_model, num_heads=num_heads)

    x = torch.randn(size=(batch_size, 4, H_img//8, W_img//8))
    time = torch.randn(size=(batch_size, 1, n_times))
    context = torch.randn(size=(batch_size, seq_len, d_context))
    output = unet(x, context, time)
    print(output.shape)


if __name__ == "__main__":
    main()