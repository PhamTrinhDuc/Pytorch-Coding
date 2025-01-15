import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, seq_len]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 vocab_size: int, 
                 max_length: int):
        super().__init__()
        self.embed_token_model = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim
        )

        self.embed_pos_model = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x: torch.Tensor):
        N, seq_len = x.size() # [B, seq_len]
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device) # [N, seq_len]
        token_embed = self.embed_token_model(x) # [N, seq_len, embed_dim]
        position_embed = self.embed_pos_model(positions) # [N, seq_len, embed_dim]
        return token_embed + position_embed # [N, seq_len, embed_dim]
    

class MultilHeadAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int):
        # d_model --> embed dimension 
        # n_heads --> number of heads 

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.WQ = nn.Linear(in_features=d_model, 
                            out_features=d_model)
        
        self.WK = nn.Linear(in_features=d_model, 
                            out_features=d_model)
        
        self.WV = nn.Linear(in_features=d_model, 
                            out_features=d_model)
        
        self.WO = nn.Linear(in_features=d_model, 
                            out_features=d_model)

    def split_heads(self, x: torch.Tensor):
        # s
        N, seq_len, d_model = x.size()
        # x: [N, seq_len, d_model] => [N, num_heads, seq_len, head_dim]
        return x.view(N, self.num_heads, seq_len, self.head_dim)
    
    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     mask=None):
        matmul_QK = torch.matmul(Q, K.transpose(-2, -1)) # [B, num_heads, seq_len, seq_len]
        dk = torch.tensor(K.size(-1), dtype=torch.float32)

        scaled_dot_product = matmul_QK / dk # [B, num_heads, seq_len, seq_len]
        if mask is not None:
            scaled_dot_product.masked_fill_(mask=mask, value=-torch.inf) # [B, num_heads, seq_len, seq_len]

        attention_scores = nn.functional.softmax(scaled_dot_product, dim=-1) # [B, num_heads, seq_len, seq_len]
        
        # [B, num_heads, seq_len, seq_len] @ [B, num_heads, seq_len, head_dim] = [b, num_heads, seq_len, head_dim]
        outputs = torch.matmul(attention_scores, V)
        return outputs, attention_scores

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):

        batch_size, seq_len, d_model = x.size()
        # [B, seq_len, d_model] => [B, seq_len, qqkv_dim] => [B, num_heads, seq_len, ]
        # divide Q, K, V by the heads after giving the classes Wk, Wq, Wo
        Q = self.split_heads(self.WQ(x)) # [B, num_heads, seq_leB, head_dim]
        K = self.split_heads(self.WK(x)) # [B, num_heads, seq_leB, head_dim]
        V = self.split_heads(self.WO(x)) # [B, num_heads, seq_leB, head_dim]

        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()[:seq_len, :seq_len]

        outputs, attention_scores = self.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return outputs # [B, seq_len, d_model]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()

        self.ffn_layers = nn.Sequential(
            nn.Linear(in_features=d_model, 
                      out_features=ff_dim), 
            nn.GELU(), 
            nn.Linear(in_features=ff_dim, 
                      out_features=d_model)
        )
    def forward(self, x):
        # x: [B, seq_len, d_model]
        return self.ffn_layers(x)
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, drop_rate: int):
        super().__init__()

        self.mha = MultilHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=drop_rate)
    
    def forward(self, x, mask=None):
        # x: [N, seq_len, embed_dim]
        # Shortcut connection for attention block
        shortcut = x
        x = self.layernorm1(x)
        x = self.mha(x, mask)
        x = self.dropout(x)
        x = shortcut + x

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut
        return x # [N, seq_len, embed_dim]