import torch 
import torch.nn as nn
from dataclasses import dataclass, asdict
from transformer import TransformerEncoder, PositionalEmbedding

@dataclass
class ViTConfig:
    d_model: int = 32
    n_heads: int = 4 
    n_layers: int = 8
    ff_dim: int = 4 * d_model
    drop_rate: float = 0.1
    image_size: tuple = (128, 128)
    patch_size: tuple = (8, 8) 
    n_channels: int = 3
    embedding_image: int = 128


class ViTEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int, 
                 n_layers: int,
                 ff_dim: int,
                 drop_rate: float,
                 image_size: tuple[int, int], 
                 patch_size: tuple[int, int], 
                 n_channels: int, 
                 embedding_image: int):
        
        super().__init__()
        assert image_size[0] % patch_size[0] == 0 or  \
            image_size[1] % patch_size[1] == 0, "image dimensions should be divisible by patch dim"
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.num_patches = (image_size[0] * image_size[1]) // (patch_size[0] * patch_size[1]) # ~ seq_len
        self.seq_len_patch = self.num_patches + 1 # add token CLS to head
        
        self.linear_proj = nn.Conv2d(in_channels=n_channels, 
                                     out_channels=d_model, 
                                     kernel_size=patch_size[0], 
                                     stride=patch_size[0])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.positions_embedding = PositionalEmbedding(d_model=d_model, 
                                                       max_seq_length=self.seq_len_patch)
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model=d_model, num_heads=n_heads, 
                               ff_dim=ff_dim, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])
        self.projection = nn.Parameter(torch.randn(d_model, embedding_image))

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        # [B, C, H, W] => (B, d_model, H//patch_size[0], W//patch_size[1])
        x = self.linear_proj(x)
        # (B, d_model, H//patch_size[0], W//patch_size[1]) => (B, d_model, dim2*dim3) => (B, dim2*dim3, d_model)
        # flatten: flatten patch to sequence, 1 pixel ! 1 token => transpose: dim2*dim3 match seq_len
        x = x.flatten(2).transpose(-2, -1) # (B, seq_patch, d_model)

        # add cls token at the beginning of patch_sequence 
        # [1, 1, d_model] => [B, 1, d_model] concat [B, seq_patch, d_model] => [B, seq_patch+1, d_model]
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1) # [B, self.seq_len_patch, d_model]
        
        x = self.positions_embedding(x) # [B, self.seq_len_patch, d_model]
        for layer in self.transformer_encoder_layers:
            x = layer(x, mask) # [B, self.seq_len_patch, d_model]
        
        x = x[:, 0, :] # [B, d_model], get embedding of token CLS 
        if self.projection is not None:
            # [B, d_model] @ [d_model, embedding_image] 
            x = x @ self.projection # [B, embedding_image]
        
        x = x / torch.norm(x, dim=-1, keepdim=True) # [B, embedding_image]
        return x # [B, embedding_image]


def main():
    args = ViTConfig()
    ViT = ViTEncoder(**asdict(args))

    mock_image = torch.randn(size=(1, 3, 128, 128))
    output = ViT(mock_image)
    print(output.shape)
    
if __name__ == "__main__":
    main()
