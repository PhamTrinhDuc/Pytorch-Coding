import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


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
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     padding=1)
        

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3, 
                                            padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        
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
    def __init__(self, d_model: int, num_heads: int=1, num_groups: int=32, d_context: int=768):
        super().__init__()
        self.channels= d_model * num_heads

        self.groupnorm_inp = nn.GroupNorm(num_channels=self.channels, 
                                          num_groups=num_groups)
        self.conv_inp = nn.Conv2d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=3, 
                                  padding=1)
        
        self.layernorm1 = nn.LayerNorm(normalized_shape=self.channels)
        self.self_attn = SelfAttention(num_heads=num_heads, 
                                       d_model=d_model)
        
        self.layernorm2 = nn.LayerNorm(normalized_shape=self.channels)
        self.cross_attn = CrossAttention(num_heads=num_heads, 
                                         dim_Q=d_model, 
                                         dim_KV=d_context)
        self.layernorm3 = nn.LayerNorm(normalized_shape=self.channels)

        self.linear1 = nn.Linear(in_features=self.channels, 
                                 out_features=self.channels * 4 * 2)
        self.linear2 = nn.Linear(in_features=self.channels * 4, 
                                 out_features=self.channels)
        
        self.conv_output = nn.Conv2d(in_channels=self.channels,
                                     out_channels=self.channels,
                                     kernel_size=3, 
                                     padding=1)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor):
        # shape x: [B, channels, H, W]
        # shape context: [B, seq_len_KV, dim_KV]
        x_origin = x

        B, channels, H, W = x.size()
        # [B, channels, H, W] => [B, channels, H, W]
        x = self.groupnorm_inp(x)
        # [B, channels, H, W] => [B, channels, H, W]
        x = self.conv_inp(x)
        # [B, channels, H, W] => [B, channels, H*W] => [B, H*W, channels]
        x = x.view((B, channels, H*W)).transpose(-2, -1)

        # Normalization + self-attention with skip connection
        shortcut = x
        # [B, H*W, channels] => [B, H*W, channels]
        x = self.layernorm1(x)
        # [B, H*W, channels] => [B, H*W, channels]
        x = self.self_attn(x)
        # [B, H*W, channels] => [B, H*W, channels]
        x = x + shortcut 

        # Normlization + cross-attention with skip connection
        shortcut = x
        # [B, H*W, channels] => [B, H*W, channels]
        x = self.layernorm2(x)
        # [B, H*W, channels] => [B, H*W, channels]
        x = self.cross_attn(x, context)
        # [B, H*W, channels] => [B, H*W, channels]
        x = x + shortcut

        # Normalizatiom+ FFN with GeLU and skip connection
        shortcut = x
        # [B, H*W, channels] => [B, H*W, channels]
        x = self.layernorm3(x)
        # [B, H*W, channels] => [B, H*W, channels*4*2]
        x = self.linear1(x)
        # [B, H*W, channels] => 2 tensors [B, H*W, channels*4]
        x, gate = x.chunk(2, dim=-1)
        # [B, H*W, channels*4] * [B, H*W, channels*4] = [B, H*W, channels*4]
        x = x * F.gelu(gate)
        # [B, H*W, channels*4] => [B, H*W, channels]
        x = self.linear2(x)
        # [B, H*W, channels] => [B, H*W, channels]
        x = x + shortcut

        # [B, H*W, channels] => [B, channels, H*W] => [B, channels, H, W]
        x = x.transpose(-2, -1).view(B, channels, H, W)
        # [B, channels, H, W] => [B, channels, H, W]
        x = self.conv_output(x)
        # [B, channels, H, W] + [B, channels, H, W] = [B, channels, H, W]
        x = x + x_origin
        return x # [B, channels, H, W]


def main():
    batch_size = 1
    in_channels = 32
    out_channels = 64
    num_groups = 32

    residuals = UnetResidualBlock(in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    num_groups=num_groups)
    mock_data= torch.randn(size=(batch_size, in_channels, 224, 224))
    time = torch.randn(size=(batch_size, 1, 1280))
    output_residual = residuals(mock_data, time)
    print("shape output residuals: ", output_residual.shape)

    # ==============================================================

    batch_size = 1
    H_img = W_img = 32
    d_model = 768
    seq_len = 1024
    num_heads=1
    num_groups=32

    x = torch.randn(size=(batch_size, d_model * num_heads, H_img, W_img))
    context = torch.randn(size=(batch_size, seq_len, d_model))
    model = UnetAttentionBlock(d_model=d_model, num_heads=num_heads, num_groups=num_groups)
    output_attention = model(x, context)
    print("shape output attention: ", output_attention.shape)


if __name__ == "__main__":
    main()
