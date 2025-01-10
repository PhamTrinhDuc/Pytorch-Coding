import torch
import math
import torch.nn as  nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    d_model: int = 2048 # origin: 4096
    n_layers: int = 8 # origin: 32
    num_heads: int = 8 # origin: 32
    head_dim: int = d_model // num_heads # depth
    n_kv_heads: Optional[int] = None
    vocab_size: int  = -1 
    ff_dim: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 0.01
    
    batch_size: int = 32
    seq_len: int = 1024 # origin: 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, 
                 depth: int, # d_model // num_heads
                 base: int = 10000):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, depth, 2) / depth)) # [depth//2 ,]
        self.register_buffer(name="theta", tensor=theta)
        
    def forward(self, x: torch.Tensor):
        # x: [N, num_heads, seq_len, depth] 
        # do seq_len kh cố định < tokenizer của gpt2 thiết kế cho chiều dài đông>, 
        # vì vậy ta sẽ tạo postions khi có seq_len đưuọc lấy từ x
        seq_len = x.size(2)
        
        positions = torch.arange(0, seq_len)[:, None] # [seq_len, 1]
        # print("positions: ", positions.shape)
        # print("theta: ", self.theta.shape)
        theta_cp = positions * self.theta
        x_even, x_odd = x[..., 0::2], x[..., 1::2] # [N, num_heads, seq_len, depth//2]
        
        sin_angles = torch.sin(theta_cp) # [seq_len, depth//2]
        cos_angles = torch.cos(theta_cp) # [seq_len, depth//2]
        # print(x_even.shape)
        # print(cos_angles.shape)

        x_even_rotated = x_even * cos_angles - x_odd * sin_angles # [N, num_heads, seq_len, depth//2]
        x_odd_rotated = x_even * sin_angles + x_odd * cos_angles # [N, num_heads, seq_len, depth//2]

        x_rotated = torch.ones_like(x) # [N, num_heads, seq_len, depth]
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(self.d_model))
    
    def forward(self, x):
        x = x.float() # [N, seq_len, d_model]
        norm = torch.sqrt(torch.mean(input=x**2, dim=-1, keepdim=True)) + self.eps
        x_hat = x / norm
        y = x_hat * self.weights
        return y # # [N, seq_len, d_model]
    

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        
        super().__init__()
        self.batch_size = args.batch_size
        self.num_heads = args.num_heads
        # Indicates the number of heads for the Keys and Values (heads in a group)
        self.n_kv_heads = args.num_heads if args.num_heads is not None else args.n_kv_heads
        # Indicates the number of heads for the Queries = num_heads
        self.n_q_heads = args.num_heads
        # Indicates how many times the Keys and Values should be repeated (num of groups)
        self.n_rep = self.n_q_heads // self.n_kv_heads
        #  Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.d_model // args.num_heads

        self.rotary_embedding = RotaryPositionalEmbedding(depth=self.head_dim)
        
        self.Wq = nn.Linear(in_features=args.d_model, 
                            out_features=self.head_dim * args.num_heads, 
                            bias=False)
        self.Wk = nn.Linear(in_features=self.d_model, 
                            out_features=self.head_dim * self.n_kv_heads,
                            bias=False)
        self.Wv = nn.Linear(in_features=args.d_model, 
                            out_features=self.head_dim * self.n_kv_heads, 
                            bias=False)
        self.Wo = nn.Linear(in_features=self.head_dim * args.num_heads, 
                            out_features=args.d_model,
                            bias=False)
        self.cache_k = torch.zeros((args.batch_size, args.seq_len, args.num_heads, args.head_dim))
        self.cache_v = torch.zeros((args.batch_size, args.seq_len, args.num_heads, args.head_dim))

    def split_heads(self, x: torch.Tensor, num_heads: int):
        # [N, seq_len, num_heads(q or kv) * head_dim] -> [N, seq_len, num_heads(q or kv), head_dim] 
        out = x.view(self.batch_size, self.args.seq_len, num_heads, self.args.head_dim)
        return out
    
    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, cache_len, num_heads_kv, head_dim = x.size() 

        if n_rep == 1:
            return x # # [N, cache_len + seq_len, num_heads_kv, head_dim]
        
        return (
            # [N, cache_len + seq_len, num_heads_kv, head_dim]
            x[:, :, :, None, :] # [N, cache_len + seq_len, num_heads_kv, 1, head_dim]
            .expand(batch_size, cache_len, num_heads_kv, n_rep, head_dim) # [N, cache_len + seq_len, num_heads_kv, n_rep, head_dim]
            .reshape(batch_size, cache_len, num_heads_kv * n_rep, head_dim) # # [N, cache_len + seq_len, num_heads_kv*n_rep, head_dim]
        )
    
    def forward(self, x: torch.Tensor, st_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, d_model = x.size() ##### seq_len = 1 ~ 1 token
        # [N, seq_len, d_model] ==> [N, seq_len, num_heads(q or kv) * head_dim] ==> [N, seq_len, num_heads(q or kv), head_dim]
        x_q = self.split_heads(x=self.Wq(x), num_heads=self.n_q_heads)  
        x_k = self.split_heads(x=self.Wk(x), num_heads=self.n_kv_heads)  
        x_v = self.split_heads(x=self.Wv(x), num_heads=self.n_kv_heads)  

        x_q = self.rotary_embedding(x_q) # [N, seq_len, num_heads_q, head_dim]
        x_k = self.rotary_embedding(x_k) # [N, seq_len, num_heads_kv, head_dim]

        # save current K, V to cache
        self.cache_k[:batch, st_pos: st_pos + seq_len, ...] = x_k # st_pos: st_pos + seq_len ~ cache_len
        self.cache_v[:batch, st_pos: st_pos + seq_len, ...] = x_v # st_pos: st_pos + seq_len ~ cache_len
        
        # Get the key and value from the beginning to the current token
        keys = self.cache_k[:batch, :st_pos + seq_len, ...] # [N, cache_len + seq_len, num_heads_kv, head_dim]
        values = self.cache_v[:batch, :st_pos + seq_len, ...] # [N, cache_len + seq_len, num_heads_kv, head_dim]

        # Repeat the number of heads of K, V (number of heads in a group) n_rep times (number of groups) to match the number of heads Q(num_heas of the model)
        keys = self.repeat_kv(keys, self.n_rep) # [N, cache_len + seq_len, num_heads_q, head_dim]
        values = self.repeat_kv(values, self.n_rep) # [N, cache_len + seq_len, num_heads_q, head_dim]


        x_q = x_q.transpose(1, 2) # [N, num_heads_q, seq_len, head_dim]
        keys = keys.transpose(1, 2) # [N, num_heads_q, cache_len + seq_len, head_dim]
        values = values.transpose(1, 2) # [N, num_heads_q, cache_len + seq_len, head_dim]

        # =============== Scaled dot product
        # [N, num_heads_q, seq_len, head_dim] @ [N, num_heads_q, head_dim, cache_len + seq_len] = [N, num_heads_q, seq_len, cache_len + seq_len]
        Y = torch.matmul(x_q, keys.transpose(2, 3)) / torch.sqrt(self.head_dim)
        if mask is not None:
            Y = Y + mask # [N, num_heads_q, seq_len, cache_len + seq_len]
        Y = nn.functional.softmax(A.float(), dim=-1).type_as(x) # [N, num_heads_q, seq_len, cache_len + seq_len]

        # [N, num_heads_q, seq_len, cache_len + seq_len] @ [N, num_heads_q, cache_len + seq_len, head_dim] = [N, num_heads_q, seq_len, head_dim]
        A = torch.matmul(Y, values)
        # [N, num_heads_q, seq_len, head_dim] ==> [N, seq_len, num_heads_q, head_dim] ==> [N, seq_len, d_model]
        A = x.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        output = self.Wo(A) # [N, seq_len, d_model]
        return output


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):

        super().__init__()
        hidden_dim = 4 * args.d_model
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # The above lines are the configuration of llama2 to help increase gpu performance
        self.linear1 = nn.Linear(in_features=args.d_model, out_features=hidden_dim, bias=False)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=args.d_model, bias=False)
        self.linear3 = nn.Linear(in_features=args.d_model, out_features=hidden_dim, bias=False)

    def forward(self, x):
        x = x.float() # [N, seq_len, d_model] 
        out1 = self.linear1(x) # [N, seq_len, hidden_dim]
        swilu = nn.functional.silu(out1) # [N, seq_len, hidden_dim]
        out2 = self.linear3(x) # [N, seq_len, hidden_dim]
        out = swilu * out2 # [N, seq_len, hidden_dim]
        out = self.linear2(out) # [N, seq_len, d_model]
        return out


class Llama2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = RMSNorm(d_model=args.d_model, eps=args.norm_eps)
        self.attention = SelfAttention(args=args)
        self.ffn = FeedForward(args=args)

    def forward(self, x: torch.Tensor, st_pos: int):
        x_norm = self.norm(x) # [N, seq_len, d_model]
        # [N, seq_len, d_model] + [N, seq_len, d_model] = [N, seq_len, d_model]
        x_attention = x + self.attention(x_norm, st_pos) 
        # [N, seq_len, d_model] + [N, seq_len, d_model] = [N, seq_len, d_model]
        out = x_attention + self.ffn(self.norm(x_attention)) 
        return out
    

class Llama2Layer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.embed_model = nn.Embedding(num_embeddings=self.vocab_size, 
                                        embedding_dim=args.d_model)
        
        self.layers = nn.ModuleList([
            Llama2Block(args) for _ in range(self.n_layers)
        ])

        self.norm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.fc = nn.Linear(in_features=args.d_model, out_features=self.vocab_size)

    def forward(self, x: torch.Tensor, st_pos: int):
        batch_size, seq_len = x.size()
        assert seq_len == 1, "Only one token at a time can be processed"
        output = self.embed_model(x) # [N, seq_len, d_model]

        freqs_complex = self.freq_complex[st_pos: st_pos + seq_len]

        for layer in self.layers:
            output = layer(output, st_pos, freqs_complex)
        output = self.norm(output)
        output = self.fc(output)
        return output.float()


def main():

    # =================================== Debugging
    args = ModelArgs

    # =========================== Rotary embedding
    mock_data = torch.randint(low=0, high=1, size=(args.batch_size, 
                                                   args.num_heads, 
                                                   args.seq_len,
                                                   args.head_dim))
    embed = RotaryPositionalEmbedding(
        depth=args.d_model // args.num_heads
    )
    freqs = embed(mock_data)
    # print(freqs.shape)

    # ============================ RMSNorm 
    mock_data = torch.randint(0, 1, size=(args.batch_size, args.seq_len, args.d_model))
    norm = RMSNorm(d_model=args.d_model)
    out_norm = norm(mock_data)
    # print(out_norm.shape)

    # ============================ Feed Forward
    mock_data = torch.randint(0, 1, (args.batch_size, args.seq_len, args.d_model))
    ffn = FeedForward(args=args)
    out_ffn = ffn(mock_data)
    print(out_ffn.shape)

if __name__ == "__main__":
    main()