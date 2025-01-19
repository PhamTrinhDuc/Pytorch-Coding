import torch
import torch.nn.functional as F
from torch import nn
from attention import SelfAttention, CrossAttention

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

class UnetResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_groups: int = 32,
                 n_times: int=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_feature = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      padding=1)
        self.linear_time = nn.Linear(in_features=n_times,
                                     out_features=out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(num_groups=num_groups,
                                             num_channels=out_channels)
        self.conv_merged = nn.Conv2d(in_channels=out_channels, 
                                     out_channels=out_channels)
        

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3, 
                                            padding=1)
    
    def forward(self, x: torch.Tensor):
        
        # x: [B, in_channels, H, W]
        # time: [1, n_time]

        shortcut = x
        # [B, in_channels, H, W] => [B, in_channels, H, W]
        x = F.silu(x)
        # [B, in_channels, H, W] => [B, in_channels, H, W
        x = self.groupnorm_feature(x)
        # [B, in_channels, H, W] => [B, out_channels, H, W]
        x = self.conv_feature(x)
        
        # [B, 1, n_times] => [B, 1, n_times]
        time = F.silu(time)
        # [B, 1, n_times] => [B, 1, out_channels]
        time = self.linear_time(time) 
        # [B, 1, out_channels] => [B, out_channels, 1] => [B, out_channels, 1, 1]
        time = time.transpose(-2, -1)[..., None]
        # [B, out_channels, H, W] +  [B, out_channels, 1, 1] =  [B, out_channels, H, W]
        merged = x + time
        # [B, out_channels, H, W] => [B, out_channels, H, W]
        merged = self.groupnorm_merged(merged)
        # [B, out_channels, H, W] => [B, out_channels, H, W]
        merged = self.conv_merged(merged)
        # [B, in_channels, H, W] => [B, out_channels, H, W]
        shortcut = self.residual_layer(shortcut)
        # [B, out_channels, H, W] + [B, out_channels, H, W] = [B, out_channels, H, W]
        merged_shortcut = merged + shortcut
        return merged_shortcut # [B, out_channels, H, W]
    

class UnetAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int=1, num_groups: int=32):
        super().__init__()
        self.groupnorm_inp = nn.GroupNorm(num_channels=None, 
                                          num_groups=num_groups)
        self.conv_inp = nn.Conv2d(in_channels=None,
                                  out_channels=None,
                                  kernel_size=3, 
                                  padding=1)
        
        self.layernorm1 = nn.LayerNorm(normalized_shape=None)
        self.self_attn = SelfAttention(num_heads=num_heads, 
                                       d_model=d_model)
        
        self.layernorm2 = nn.LayerNorm(normalized_shape=None)
        self.cross_attn = CrossAttention(num_heads=num_heads, 
                                         dim_Q=d_model, 
                                         dim_KV=d_model)
        self.layernorm3 = nn.LayerNorm(normalized_shape=None)

        self.linear1 = nn.Linear(in_features=None, 
                                 out_features=None)
        self.linear2 = nn.Linear(in_features=None, 
                                 out_features=None)
        
        self.conv_output = nn.Conv2d(in_channels=None,
                                out_channels=None,
                                kernel_size=3, 
                                padding=1)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # shape x: [B, C, H, W]
        # shape context: [B, seq_len_KV, dim_KV]
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