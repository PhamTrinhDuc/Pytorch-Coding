import torch
import torch.nn as nn
import torch.nn.functional as F
# from attention import SelfAttention

class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wo = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, d_model = x.size()
        return x.view(B, self.num_heads, seq_len, self.head_dim)
    
    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     causal_mask: bool=False):
        
        matmul_QK = torch.matmul(Q, K.transpose(-2, -1)) # [B, num_heads, seq_len, seq_len]

        scaled_dot_product = matmul_QK / self.head_dim # [B, num_heads, seq_len, seq_len]
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(scaled_dot_product, dtype=torch.bool).triu(diagonal=1)
            # Fill the upper triangle with -inf
            scaled_dot_product.masked_fill_(mask, value=-torch.inf)
        
        attentions_scores = F.softmax(input=scaled_dot_product, dim=-1) # [B, num_heads, seq_len, seq_len]
        
        # [B, num_heads, seq_len, seq_len] @ [B, num_heads, seq_len, head_dim] = [B, num_heads, seq_len, head_dim]
        output = torch.matmul(attentions_scores, V)
        return output, attentions_scores # [B, num_heads, seq_len, seq_len]


    def forward(self, x: torch.Tensor, causal_mask: bool=False) -> torch.Tensor:
        # shape x: [B, seq_len, d_model]

        # [B, seq_len, d_model] => [B, seq_len, d_model] => [B, num_heads, seq_len, head_dim]
        Q = self.split_heads(self.Wq(x)) 
        K = self.split_heads(self.Wk(x))
        V = self.split_heads(self.Wv(x))

        output, attention_scores = self.scaled_dot_product_attention(Q=Q, K=K, V=V, causal_mask=causal_mask)

        output = output.transpose(1, 2).contiguous().view(*x.size()) # [B, seq_len, d_model]
        output = self.Wo(output) # [B, seq_len, d_model]
        return output # [B, seq_len, d_model]
    

class VAEResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 output_channels: int,
                 num_groupnorm: int = 32, 
                 kernel_size: int = 3, 
                 padding: int = 1, 
                 stride: int = 1, 
                 dilation: int = 1):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(num_groups=num_groupnorm, num_channels=in_channels)          

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=output_channels,
                               kernel_size=kernel_size, 
                               padding=padding, 
                               stride=stride, 
                               dilation=dilation)
        
        self.groupnorm2 = nn.GroupNorm(num_groups=num_groupnorm, num_channels=output_channels)          
        self.conv2 = nn.Conv2d(in_channels=output_channels, 
                               out_channels=output_channels,
                               kernel_size=kernel_size,
                               padding=padding, 
                               stride=stride, 
                               dilation=dilation)
        
        if in_channels == output_channels:
            self.layer_residual = nn.Identity()
        else:
            self.layer_residual = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=output_channels,
                                            kernel_size=kernel_size,
                                            padding=padding, 
                                            stride=stride, 
                                            dilation=dilation)
        
    def forward(self, x: torch.Tensor):
        # shape x: [B, C, H, W]
        residual = x
        x = self.groupnorm1(x) # [B, C, H, W]
        x = F.silu(x)
        x = self.conv1(x) # [B, out_channels, H, W]

        x = self.groupnorm2(x) # [B, out_channels, H, W]
        x = F.silu(x) # [B, out_channels, H, W]
        x = self.conv2(x) # [B, out_channels, H, W]

        # [B, out_channels, H, W] + [B, out_channels, H, W] = [B, out_channels, H, W]
        x = x + self.layer_residual(residual) # [B, out_channels, H, W]
        return x # [B, out_channels, H, W]


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 2, num_groupnorms: int=32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_channels=channels, num_groups=num_groupnorms)
        self.attention = SelfAttention(num_heads=num_heads, d_model=channels)

    def forward(self, x: torch.Tensor):
        # shape x: [B, C, H, W]
        residual = x
        x = self.groupnorm(x) # [B, C, H, W]
        B, C, H, W = x.size()

        # [B, C, H, W] => [B, C, H*W] => [B, H*W, C] ~ [B, seq_len, d_model]
        x = x.view(B, C, H * W).transpose(dim0=-2, dim1=-1)

        x = self.attention(x) # [B, H*W, C]
        # [B, H*W, C] => [B, C, H * W] => [B, C, H, W]
        x = x.transpose(-2, -1).view(B, C, H, W)

        x = x + residual # [B, C, H, W]
        return x # [B, C, H, W]

def main():
    mock_data = torch.randn(size=(1, 32, 128, 128))
    residual_block = VAEResidualBlock(in_channels=32, output_channels=64)


    attention_block = VAEAttentionBlock(channels=32)

    output_residual = residual_block(mock_data)
    print("Shape output residual: ", output_residual.shape)
    output_attention= attention_block(mock_data)
    print("Shape output attention: ", output_attention.shape)

if __name__ == "__main__":
    main()
