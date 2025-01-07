import math
import torch 
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)

        return x


class TransformersModel(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int,
                 n_heads: int, 
                 hidden_dims: int, 
                 n_layers: int, 
                 dropout: float =0.5):
        
        super(TransformersModel, self).__init__()
        self.model_type = "Transformer"
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim)

        self.pos_encoder = PositionalEncoding(embedding_dim=embedding_dim, 
                                              dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads, 
            dim_feedforward=hidden_dims, 
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=n_layers
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        self.init_weights()


    def init_weights(self):
        initarange = 0.1
        self.embedding.weight.data.uniform_(-initarange, initarange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initarange, initarange)


    def forward(self, src, src_mask=None, padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)

        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(DEVICE)
        
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask=padding_mask)
        output = self.linear(output)

        return output


        
        