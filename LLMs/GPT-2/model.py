import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass


@dataclass
class GPTConfig124M:
    vocab_size: int = 50257
    seq_len: int = 1024
    d_model:int =  768
    ff_dim: int = d_model * 4
    max_new_tokens: int = 10
    num_heads: int = 12
    num_layers: int = 12
    drop_rate:float =  0.05
    qkv_bias: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 vocab_size: int, 
                 max_length: int, 
                 device: str):
        super().__init__()
        self.device = device
        self.embed_model = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim
        )

        self.pos_embed = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(device=self.device) # [N, seq_len]
        token_embed = self.embed_model(x) # [N, seq_len, embed_dim]
        position_embed = self.pos_embed(positions) # [N, seq_len, embed_dim]
        return token_embed + position_embed # [N, seq_len, embed_dim]


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        # x: [N, seq_len, embed_dim]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (torch.sqrt(var + self.eps))
        return x_norm * self.scale + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, dropout: float = 0.05,):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wo = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)


    def split_heads(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, self.num_heads, seq_len, self.head_dim)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask = None):
        matmul_QK = torch.matmul(Q, K.transpose(2, 3)) # [N, num_heads, seq_len, seq_len]
        dk = torch.tensor(K.size(-1), dtype=torch.float32)
        scaled_dot_product = matmul_QK / dk # [N, num_heads, seq_len, seq_len]
        # masked 
        if mask is not None:
            scaled_dot_product.masked_fill_(mask=mask, value=-torch.inf) 

        attention_weights = F.softmax(scaled_dot_product, dim=-1) # [N, num_heads, seq_len, seq_len]
        # [N, num_heads, seq_len, seq_len] @  [N, num_heads, seq_len, head_dim] = [N, num_heads, seq_len, head_dim]
        output = torch.matmul(attention_weights, V)
        return attention_weights, output

    def forward(self, x: torch.Tensor,):
        # x: [N, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        # [N, seq_len, d_model] => [N, seq_len, d_model] => [N, num_heads, seq_len, head_dim]
        Q = self.split_heads(self.Wq(x)) 
        K = self.split_heads(self.Wk(x))
        V = self.split_heads(self.Wv(x))

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()[:seq_len, :seq_len]

        attention_weights, output = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        # [N, num_heads, seq_len, head_dim] => [N, seq_len, num_heads, head_dim] => [N, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        A = self.Wo(output)
        return A # [N, seq_len, d_model]


class FeedForward(nn.Module):
    def __init__(self, args: GPTConfig124M):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=args.d_model, out_features=args.ff_dim), 
            GELU(),
            nn.Linear(in_features=args.ff_dim, out_features=args.d_model)
        )

    def forward(self, x: torch.Tensor):
        # x: [N, seq_len, d_model]
        return self.ffn(x)

class TransformerBlock(nn.Module):
    def __init__(self, args: GPTConfig124M):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=args.d_model, 
            num_heads=args.num_heads, 
            dropout=args.drop_rate, 
            seq_len=args.seq_len
        )
        self.ffn = FeedForward(args=args)
        self.norm1 = LayerNorm(embed_dim=args.d_model)
        self.norm2 = LayerNorm(embed_dim=args.d_model)
        self.dropout = nn.Dropout(p=args.drop_rate)
    
    def forward(self, x):
        # x: [N, seq_len, embed_dim]
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, args: GPTConfig124M):
        super().__init__()
        self.args = args
        self.embedding = TokenAndPositionEmbedding(embed_dim=args.d_model, 
                                                   vocab_size=args.vocab_size,
                                                   max_length=args.seq_len,
                                                   device=args.device)
        self.drop_embed = nn.Dropout(p=args.drop_rate)
        
        self.layers = nn.ModuleList([
            TransformerBlock(args=args) for _ in range(args.num_layers)
        ])

        self.norm = LayerNorm(embed_dim=args.d_model)
        self.fc = nn.Linear(in_features=args.d_model, out_features=args.vocab_size)

    def forward(self, x):
        # x: [N, seq_len]
        embedding = self.embedding(x) # [N, seq_len, d_model]
        embed_dropout = self.drop_embed(embedding) # [N, seq_len, d_model]

        for layer in self.layers:
            logits = layer(embed_dropout) # # [N, seq_len, d_model]
        logits = self.norm(logits) # [N, seq_len, d_model]
        logits = self.fc(logits) # [N, seq_len, vocab_size]
        return logits

def generate_text_simple(model: GPTModel, 
                         input: torch.Tensor, 
                         max_new_tokens: int, 
                         context_length: int):
    
    for _ in range(max_new_tokens):
        input_model= input[:, -context_length:] # [N, seq_len]
        # get predictions
        with torch.no_grad():
            logits = model(input_model)
        
        # Focus only on the last time step
        # (batch, seq_len, vocab_size) => (batch, vocab_size)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # [batch, 1]
        input = torch.cat((input, idx_next), dim=1) # [batch, seq_len+1]
    return input # [batch, len(context_tokens) + max_new_tokens]



def main():
    # ================================= DEBUGGING
    # ================================= MultiHeadAttention
    args = GPTConfig124M
    # mock_data = torch.randn(size=(1, args.seq_len, args.d_model))
    # multihead = MultiHeadAttention(d_model=args.d_model, num_heads=args.num_heads)
    # output = multihead(mock_data)
    # print(output.shape)

    # ================================= Transformer Block
    # gpt_model = TransformerBlock(args=args)
    # output = gpt_model(mock_data)
    # print(output.shape)

    # ================================= GPT2 Model
    # mock_data = torch.randint(0, 10, size=(1, args.seq_len))
    # gpt2_model = GPTModel(args=args)
    # logits = gpt2_model(mock_data)
    # print(logits.shape)

    # ================================= generate text
    
    torch.manual_seed(123)
    model = GPTModel(args=args)
    model.eval()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    encoded_text = torch.tensor(tokenizer.encode(text=start_context)).unsqueeze(0)

    print("Origin tokens: ", encoded_text)
    print("Origin text: ", start_context)

    output = generate_text_simple(
        model = model,
        input=encoded_text, 
        max_new_tokens=args.max_new_tokens,
        context_length=args.seq_len
    )
    decoded_text = tokenizer.decode(tokens=output.squeeze(0).tolist())
    print("New tokens: ", output)
    print("New text: ", decoded_text)

# coding gpt2 from scratch
if __name__ == "__main__":
    main()