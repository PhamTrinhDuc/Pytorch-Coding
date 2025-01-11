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
    num_heads: int = 12
    num_layers: int = 12
    drop_rate:float =  0.05
    qkv_bias: bool = False


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_len: int, stride: int):
        super().__init__()
        self.inputs_ids = []
        self.target_ids = []

        # tokenize the entire text 
        input_ids = tokenizer.encode(text, allowed_special = {"<endoftext>"})
        for i in range(0, len(input_ids) - max_len, stride):
            input_chunk = input_ids[i: i + max_len]
            target_chunk = input_ids[i+1: i + max_len + 1]
            self.inputs_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.inputs_ids)
    
    def __getitem__(self, index):
        input_chunk = self.inputs_ids[index]
        target_chunk = self.target_ids[index]
        return input_chunk, target_chunk
    

def create_dataloader(text: str, batch_size: 4, max_len: int = 256, stride: int = 128, 
                      is_shuffle = True, is_drop_last = True, num_workers: int = 2):
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    dataset = GPTDataset(text=text, tokenizer=tokenizer, max_len=max_len, stride=stride)

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=is_shuffle, 
        num_workers=num_workers, 
        drop_last=is_drop_last
    )
    return dataloader


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

        self.mask =torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    def split_heads(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, self.num_heads, seq_len, self.head_dim)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        matmul_QK = torch.matmul(Q, K.transpose(2, 3)) # [N, num_heads, seq_len, seq_len]
        dk = torch.tensor(K.size(-1), dtype=torch.float32)
        scaled_dot_product = matmul_QK / dk # [N, num_heads, seq_len, seq_len]
        # masked 
        scaled_dot_product = scaled_dot_product.masked_fill_(mask=self.mask, value=-torch.inf) 

        attention_weights = F.softmax(scaled_dot_product, dim=-1) # [N, num_heads, seq_len, seq_len]
        # [N, num_heads, seq_len, seq_len] @  [N, num_heads, seq_len, head_dim] = [N, num_heads, seq_len, head_dim]
        output = torch.matmul(attention_weights, V)
        return attention_weights, output

    def forward(self, x: torch.Tensor):
        # x: [N, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        # [N, seq_len, d_model] => [N, seq_len, d_model] => [N, num_heads, seq_len, head_dim]
        Q = self.split_heads(self.Wq(x)) 
        K = self.split_heads(self.Wk(x))
        V = self.split_heads(self.Wv(x))

        attention_weights, output = self.scaled_dot_product_attention(Q, K, V)
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
        self.embedding_token = nn.Embedding(num_embeddings=args.vocab_size, 
                                            embedding_dim=args.d_model)
        self.embedding_pos = nn.Embedding(num_embeddings=args.seq_len, 
                                          embedding_dim=args.d_model)
        self.drop_embed = nn.Dropout(p=args.drop_rate)
        
        self.layers = nn.ModuleList([
            TransformerBlock(args=args) for _ in range(args.num_layers)
        ])

        self.norm = LayerNorm(embed_dim=args.d_model)
        self.fc = nn.Linear(in_features=args.d_model, out_features=args.vocab_size)

    def forward(self, x):
        # x: [N, seq_len]
        embed_token = self.embedding_token(x) # [N, seq_len, d_model]
        embed_pos = self.embedding_pos(x) # [N, seq_len, d_model]
        embed_final = embed_token + embed_pos # [N, seq_len, d_model]
        embed_dropout = self.drop_embed(embed_final) # [N, seq_len, d_model]

        for layer in self.layers:
            logits = layer(embed_dropout) # # [N, seq_len, d_model]
        logits = self.norm(logits) # [N, seq_len, d_model]
        logits = self.fc(logits) # [N, seq_len, vocab_size]
        return logits


def main():
    # ================================= DEBUGGING
    # ================================ MultiHeadAttention
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
    mock_data = torch.randint(0, 10, size=(1, args.seq_len))
    gpt2_model = GPTModel(args=args)
    logits = gpt2_model(mock_data)
    print(logits.shape)


if __name__ == "__main__":
    main()