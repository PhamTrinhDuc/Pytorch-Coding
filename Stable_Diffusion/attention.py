import torch
import math
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads != 0, "d_model must be divisible by num_heads"
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
        # [B, num_heads, seq_len, d_model] @ [B, num_heads, d_model, seq_len] = [B, num_heads, seq_len, seq_len]
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
    

class CrossAttention(nn.Module):
    def __init__(self, num_heads: int, dim_Q: int, dim_KV: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_Q // num_heads
        self.Wq = nn.Linear(in_features=dim_Q, out_features=dim_Q, bias=False)
        self.Wk = nn.Linear(in_features=dim_KV, out_features=dim_Q, bias=False)
        self.Wv = nn.Linear(in_features=dim_KV, out_features=dim_Q, bias=False)
        self.Wo = nn.Linear(in_features=dim_Q, out_features=dim_Q, bias=False)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, self.num_heads, seq_len, self.head_dim)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # [B, num_heads, seq_len_Q, dim_Q] @ [B, num_heads, dim_Q, seq_len_KV] = [B, num_heads, seq_len_Q, seq_len_KV]
        matmul_Qk = torch.matmul(Q, K.transpose(-2, -1))

        matmul_Qk = matmul_Qk / self.head_dim # [B, num_heads, seq_len_Q, seq_len_KV]

        attention__scores = F.softmax(matmul_Qk, dim=-1) # [B, num_heads, seq_len_Q, seq_len_KV]
        # [B, num_heads, seq_len_Q, seq_len_KV] @ [B, num_heads, seq_len_KV, dim_Q] = [B, num_heads, seq_len_Q, dim_Q]
        output = torch.matmul(attention__scores, V) # [B, num_heads, seq_len_Q, dim_Q]
        
        return (output, # [B, num_heads, seq_len_Q, dim_Q]
                attention__scores # [B, num_heads, seq_len_Q, seq_len_KV]
        )
    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV)

        # [B, seq_len_Q, dim_Q] => [B, seq_len_Q, dim_Q] => [B, num_heads, seq_len_Q, head_dim]
        Q = self.split_heads(self.Wq(x_q))
        # [B, seq_len_Q, dim_Q] => [B, seq_len_KV, dim_Q] => [B, num_heads, seq_len_KV, head_dim]
        K = self.split_heads(self.Wk(x_kv))
        # [B, seq_len_Q, dim_Q] => [B, seq_len_KV, dim_Q] => [B, num_heads, seq_len_KV, head_dim]
        V = self.split_heads(self.Wv(x_kv))

        output, attention_scores = self.scaled_dot_product_attention(Q, K, V)
        # [B, num_heads, seq_len_Q, dim_Q] => # [B, seq_len_Q, num_heads, dim_Q] => [B, num_heads_dim_Q]
        output = output.transpose(1, 2).contiguous().view(*x_q.size())

        # [B, num_heads_dim_Q] => [B, num_heads_dim_Q]
        output = self.Wo(output)
        return output # [B, num_heads_dim_Q]



def main():
    batch_size = 32
    seq_len = 1024
    d_model = d_cross = 768
    num_heads = 4
    self_attention = SelfAttention(d_model=d_model, num_heads=num_heads)
    cross_attention = CrossAttention(num_heads=num_heads, dim_Q=d_model, dim_KV=d_cross)

    mock_data = torch.randn(size=(batch_size, seq_len, d_model))
    output_self_attention = self_attention(mock_data)
    print("Shape output self attention: ", output_self_attention.size())
    output_cross_attention = cross_attention(mock_data, mock_data)
    print("Shape output cross attention: ", output_cross_attention.size())

if __name__ == "__main__":
    main()
