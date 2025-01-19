import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 seq_len: int, 
                 device: str = "cpu"):
        super().__init__()
        self.device = device
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, 
                                        embedding_dim=d_model)
        
        self.position_embed = nn.Embedding(num_embeddings=seq_len,
                                           embedding_dim=d_model)
        
    def forward(self, tokens: torch.LongTensor):
        # shape x: [B, seq_len]
        B, seq_len = tokens.size()
        embed_tokens = self.token_embed(tokens) # [B, seq_len, d_model] 
        # [1, seq_len] => [B, seq_len]
        positions = torch.arange(0, seq_len).expand(B, seq_len).to(device=self.device) 
        # [B, seq_len, d_model]
        embed_positions = self.position_embed(positions)
        # [B, seq_len, d_model] + [B, seq_len, d_model] = [B, seq_len, d_model]
        return embed_tokens + embed_positions
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model*4),
            nn.GELU(),
            nn.Linear(in_features=d_model*4, out_features=d_model),
        )

    def forward(self, x: torch.Tensor):
        # [B, seq_len, d_model] => [B, seq_len, d_model]
        output = self.layers(x)
        return output

class CLIPLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int=1, drop_rate: float=0.1):
        super().__init__()
        self.attention = SelfAttention(num_heads=num_heads, d_model=d_model)
        self.ffn = FeedForward(d_model=d_model)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=drop_rate)
    
    def forward(self, x: torch.Tensor):
        # x: [B, seq_len, d_model]
        shortcut = x # [B, seq_len, d_model]
        x = self.layernorm1(x) # [B, seq_len, d_model]
        x = self.attention(x) # [B, seq_len, d_model]
        x = self.dropout(x)
        x = x + shortcut # [B, seq_len, d_model]

        shortcut = x # [B, seq_len, d_model]
        x = self.layernorm2(x) # [B, seq_len, d_model]
        x = self.ffn(x) # [B, seq_len, d_model]
        x =self.dropout(x) # [B, seq_len, d_model]
        x = x + shortcut # [B, seq_len, d_model]
        return x # [B, seq_len, d_model]
    

class CLIP(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 seq_len: int, 
                 d_model: int, 
                 num_layers: int=1,
                 num_heads: int=1, 
                 drop_rate: float=0.1,
                 device:str="cpu"):
        super().__init__()
        self.embed_model = CLIPEmbedding(vocab_size=vocab_size, 
                                         d_model=d_model, 
                                         seq_len=seq_len,
                                         device=device)
        self.layers = nn.ModuleList([
            CLIPLayer(d_model=d_model, 
                      num_heads=num_heads,
                      drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        
        self.layernorm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, tokens: torch.LongTensor): 
        # x: [B, seq_len]
        embedding = self.embed_model(tokens) # [B, seq_len, d_model]
        for layer in self.layers:
            output = layer(embedding) #[B, seq_len, d_model]
        
        output = self.layernorm(output) # [B, seq_len, d_model]
        return output # [B, seq_len, d_model]


def main():
    vocab_size = 49408
    d_model = 768
    seq_len = 77
    num_heads = 1
    num_layers = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIP(vocab_size=vocab_size, 
                 seq_len=seq_len, 
                 d_model=d_model)
    mock_data = torch.randint(low=0, high=10, size=(32, seq_len))
    output = model(mock_data)
    print(output.shape)

if __name__ == "__main__":
    main()