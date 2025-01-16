import torch
import tiktoken
import torch.nn as nn
from dataclasses import dataclass, asdict
from transformer import TransformerEncoder, TokenAndPositionEmbedding
from config import TextEncoderConfig



def tokenization(text: str, 
                 tokenizer = None,
                 encode: bool=True, 
                 mask=None, 
                 max_seq_length=32):
    
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    if encode:
        # Encode văn bản thành tokens
        tokens = tokenizer.encode(text)
        
        # Thêm token EOS (50256) ở cuối
        # GPT-2 không sử dụng token BOS riêng biệt
        tokens = tokens + [50256]  # EOS token
        
        # Cắt ngắn nếu độ dài vượt quá max_seq_length
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            
        # Tạo tensor từ tokens
        out = torch.tensor(tokens, dtype=torch.long)
        
        # Tạo attention mask
        mask = torch.ones(len(out) + 1)
        
        # Pad tokens nếu cần thiết
        if len(out) < max_seq_length:
            padding_length = max_seq_length - len(out)
            # Pad với giá trị 0
            out = torch.cat([out, torch.zeros(padding_length, dtype=torch.long)])
            # Pad mask với giá trị 0
            mask = torch.cat([mask, torch.zeros(padding_length)])
            
        return out, mask.type(torch.IntTensor)
    
    else:
        # Decode từ tokens về text
        # Bỏ qua EOS token và các giá trị padding
        valid_tokens = text[:len(mask.nonzero())-1].tolist()  # Chỉ bỏ EOS ở cuối
        out = tokenizer.decode(valid_tokens)
        return out, None


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, 
                 max_seq_len: int, n_layers: int, n_heads: int,
                 embedding_image: int, ff_dim: int, 
                 drop_rate: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len + 1
        self.embed_model = TokenAndPositionEmbedding(embed_dim=d_model, 
                                                     vocab_size=vocab_size, 
                                                     max_length=max_seq_len)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

        self.transformer_enc_layers = nn.ModuleList([
            TransformerEncoder(d_model=d_model, num_heads=n_heads, 
                               ff_dim=ff_dim, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])

        self.projection = nn.Parameter(torch.randn(d_model, embedding_image))

    def forward(self, x: torch.LongTensor, mask: torch.Tensor=None):
        # x: [B, max_seq_len]
        embedding = self.embed_model(x) # [B, max_seq_len, d_model]

        # [1, 1, d_model] => [B, 1, d_model] concat [B, seq-len, d_model] => [B, self.max_seq_len, d_model]
        # add token cls to head embedding
        embedding = torch.cat((self.cls_token.expand(x.size(0), -1, -1), embedding), dim=1)
        
        for encoder_layer in self.transformer_enc_layers:
            output = encoder_layer(embedding, mask) # [B, self.max_seq_len, d_model]
        
        embedding_final = output[:, 0, :] # [B, d_model], get embedding token CLS
        embedding_final = embedding_final @ self.projection # [B, embedding_image]
        embedding_final = embedding_final / torch.norm(embedding_final, dim=-1, keepdim=True)
        return embedding_final # [B, embedding_image]


class TextEncoderRetrieval(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, 
                 max_seq_len: int, n_layers: int, 
                 n_heads: int, embedding_image: int, 
                 ff_dim: int, drop_rate: int):
        super().__init__()
        self.max_seq_len = max_seq_len + 1
        self.embed_model = TokenAndPositionEmbedding(embed_dim=d_model, 
                                                     vocab_size=vocab_size, 
                                                     max_length=max_seq_len)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

        self.transformer_enc_layers = nn.ModuleList([
            TransformerEncoder(d_model=d_model, num_heads=n_heads, 
                               ff_dim=ff_dim, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])

        self.projection = nn.Parameter(torch.randn(d_model, embedding_image))

    def forward(self, tokens: torch.LongTensor, mask=None):
        # tokens: [B, max_seq_len]
        embedding = self.embed_model(tokens) # [B, max_seq_len, embed_dim]

        for encoder_layer in self.transformer_enc_layers:
            output = encoder_layer(embedding, mask)
        
        if mask is not None:
            # Get the lengths of each sequence (i.e., find the last non-padded token)
            idx_seq_len = mask.sum(dim=-1) - 1 # Subtract 1 to get the index
            output = output[torch.arange(tokens.shape[0]), idx_seq_len]
        
        else:
            output = output[:, -1] # If no mask is provided, take the last token in the sequence.
        
        output = output @ self.projection
        
        output = output / torch.norm(output, dim=-1, keepdim=True)
        return output # [B, embedding_image]


def main():
    text = "Hello, I am Duc"
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    output, mask = tokenization(text=text, tokenizer=tokenizer, encode=True)
    print(output.shape)

    args = TextEncoderConfig(vocab_size=tokenizer.n_vocab)
    text_encoder = TextEncoder(**asdict(args))
    mock_text = torch.randint(low=0, high=10, size=(1, 32))
    output = text_encoder(mock_text)
    print(output.shape)

if __name__ == "__main__":
    main()
