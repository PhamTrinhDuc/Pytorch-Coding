import torch 
import torch.nn as nn

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 d_model: int=1152, 
                 image_size: int=224,
                 patch_size: int=14,):
        
        """
        (embeddings): SiglipVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
            (position_embedding): Embedding(256, 1152)
        )
        """
        super().__init__()

        self.patch_embeder = nn.Conv2d(in_channels=in_channels, 
                                       out_channels=d_model,
                                       kernel_size=patch_size,
                                       stride=patch_size,
                                       padding="valid") # no padding
        
        self.num_positions = num_patches = (image_size // patch_size) ** 2 # 16 * 16 = 256
        self.position_embeder = nn.Embedding(num_embeddings=self.num_positions, 
                                             embedding_dim=d_model)
        
    def forward(self, pixel_values: torch.Tensor):
        # [B, C, H, W] => [B, d_model, H//patch_size, W//patch_size]
        patch_embed = self.patch_embeder(pixel_values)
        # [B, d_model, H//patch_size, W//patch_size] => [B, d_model, num_patches]
        # where H//patch_size * W//patch_size = num_patches
        patch_embed = patch_embed.flatten(2)
        # [B, d_model, num_patches] => [B, num_patches, d_model] ~ [B, seq_len, d_model]
        patch_embed = patch_embed.transpose(-2, -1)
        
        # [B, num_positions]
        positions = torch.arange(0, self.num_positions, device=pixel_values.device).expand(size=(1, -1))
        # [B, num_positions] => [B, num_positions, d_model]
        position_embed = self.position_embeder(positions)
        # [B, num_positions, d_model] + [B, num_patches, d_model] = [B, num_patches, d_model]
        embedding = position_embed + patch_embed 
        return embedding


class SiglipAttention(nn.Module):
    def __init__(self, 
                 d_model: int=1152, 
                 num_heads: int=12, 
                 is_bias: bool=True):
        super().__init__()

        """
        (self_attn): SiglipAttention(
            (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
            (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
            (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
            (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
        )
        """

        assert d_model % num_heads == 0, "d_model should be divisible by n_heads"
        self.d_model = d_model # 1152
        self.num_heads = num_heads # 12 
        self.head_dim = d_model // num_heads # 96

        self.WQ = nn.Linear(in_features=d_model, out_features=d_model, bias=is_bias)
        self.WK = nn.Linear(in_features=d_model, out_features=d_model, bias=is_bias)
        self.WV = nn.Linear(in_features=d_model, out_features=d_model, bias=is_bias)
        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=is_bias)

    
    def split_heads(self, x: torch.Tensor):
        B, num_patches, d_model = x.size()
        # [B, num_patches, d_model] => [B, num_heads, num_patches, head_dim]
        return x.view(B, self.num_heads, num_patches, self.head_dim)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # [B, num_heads, num_patches, head_dim] @ [B, num_heads, head_dim, num_patches] = [B, num_heads, num_patches, num_patches]
        matmul_QK = torch.matmul(Q, K.transpose(-2, -1))
        dk = self.head_dim ** -0.5 # 1/sqrt(dk)
        # [B, num_heads, num_patches, num_patches]
        matmul_QK = matmul_QK * dk
        # [B, num_heads, num_patches, num_patches]
        attention_weights = nn.functional.softmax(matmul_QK, dim=-1)
        # [B, num_heads, num_patches, num_patches]@[B, num_heads, num_patches, head_dim] = [B, num_heads, num_patches, head_dim]
        attention_output = torch.matmul(attention_weights, V)
        return attention_output, attention_weights

    def forward(self, x: torch.Tensor):
        # shape x: [B, num_patches, d_model], output of the embedding layer

        # [B, num_patches, d_model] => [B, num_heads, num_patches, head_dim]
        Q = self.split_heads(x=self.WQ(x))
        K = self.split_heads(x=self.WK(x))
        V = self.split_heads(x=self.WV(x))
        # [B, num_heads, num_patches, head_dim] => [B, num_heads, num_patches, head_dim]
        output_attention, attention_weights = self.scaled_dot_product_attention(Q=Q, K=K, V=V)
        # [B, num_heads, num_patches, head_dim] => [B, num_patches, num_heads, head_dim] 
        output_attention = output_attention.transpose(1, 2).contiguous()
        # [B, num_patches, num_heads, head_dim]  => [B, num_patches, d_model] 
        output_attention = output_attention.view(x.shape)
        # [B, num_patches, d_model]  => [B, num_patches, d_model] 
        out_projection = self.out_proj(output_attention)
        return out_projection # [B, num_patches, d_model] 


class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input, approximate="tanh")
    

class SiglipMLP(nn.Module):
    def __init__(self, 
                 d_model: int=1152,
                 hidden_dim: int=4304):
        super().__init__()
        """
        (mlp): SiglipMLP(
            (activation_fn): PytorchGELUTanh()
            (fc1): Linear(in_features=1152, out_features=4304, bias=True)
            (fc2): Linear(in_features=4304, out_features=1152, bias=True)
        )
        """
        self.act_func = PytorchGELUTanh()
        self.fc1 = nn.Linear(in_features=d_model, 
                             out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, 
                             out_features=d_model)
    
    def forward(self, input: torch.Tensor):
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        output = self.act_func(input)
        # [B, num_patches, d_model] => [B, num_patches, hidden_dim]
        output = self.fc1(output)
        # [B, num_patches, hidden_dim] => [B, num_patches, d_model]
        output = self.fc2(output)
        return output # hidden_dim


class SiglipEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int=1152,
                 num_heads: int=12, 
                 eps: float=1e-06,
                 hidden_mlp: int=4304):
        
        super().__init__()

        """
        SiglipEncoderLayer(
            (self_attn): SiglipAttention(
                (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
            )
            (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
            (mlp): SiglipMLP(
                (activation_fn): PytorchGELUTanh()
                (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                (fc2): Linear(in_features=4304, out_features=1152, bias=True)
            )
            (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
        )
        """

        self.self_attn = SiglipAttention(d_model=d_model, num_heads=num_heads,)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.mlp = SiglipMLP(d_model=d_model, hidden_dim=hidden_mlp)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model, eps=eps)
    
    
    def forward(self, x: torch.Tensor):

        residual = x
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        out_attn = self.self_attn(x)
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        out_norm1 = self.layer_norm1(out_attn) + residual

        residual = out_norm1
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        out_mlp = self.mlp(out_norm1)
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        output = self.layer_norm2(out_mlp) + residual
        return output
    

class SiglipEncoder(nn.Module):
    def __init__(self,
                 n_layers: int=27,
                 d_model: int=1152,
                 num_heads: int=12,
                 eps: float=1e-06,
                 hidden_mlp: int=4304):
        super().__init__()
        """
        (encoder): SiglipEncoder(
            (layers): ModuleList(
                (0-26): 27 x SiglipEncoderLayer(
                    (...)
                )
            )
        )
        """
        self.layers = nn.ModuleList(modules=[SiglipEncoderLayer(d_model=d_model, 
                                                               num_heads=num_heads, 
                                                               eps=eps, 
                                                               hidden_mlp=hidden_mlp)
                                    for _ in range(n_layers)])
    def forward(self, input: torch.Tensor):
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        for layer in self.layers:
            out = layer(input)
        return out
    

class SiglipVisionTransformer(nn.Module):
    def __init__(self, 
                 in_channels: int=3,
                 d_model: int=1152,
                 num_heads: int=12,
                 n_layers: int=27,
                 hidden_mlp: int=4303,
                 image_size: int=224,
                 patch_size: int=14,
                 eps: float=1e-06):
        """
        (vision_model): SiglipVisionTransformer(
            (embeddings): SiglipVisionEmbeddings(
                (...)
            )
            (encoder): SiglipEncoder(
                (layers): ModuleList(
                    (0-26): 27 x SiglipEncoderLayer(
                        (self_attn): SiglipAttention(
                            (...)
                        )
                    )
                )
            )
            (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
        )
        """
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(in_channels=in_channels, 
                                                 d_model=d_model,
                                                 image_size=image_size,
                                                 patch_size=patch_size)
        self.encoder = SiglipEncoder(n_layers=n_layers,
                                     d_model=d_model,
                                     num_heads=num_heads,
                                     eps=eps,
                                     hidden_mlp=hidden_mlp)
        self.post_layernorm = nn.LayerNorm(normalized_shape=d_model, 
                                           eps=eps)
        
    def forward(self, input: torch.Tensor):
        # input: [B, C, H, W] = > [B, num_patches, d_model]
        embedding = self.embeddings(input)
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        out_encoder = self.encoder(embedding)
        # [B, num_patches, d_model] => [B, num_patches, d_model]
        output = self.post_layernorm(out_encoder)
        return output # [B, num_patches, d_model]


class SiglipVisionModel(nn.Module):
    def __init__(self,
                 in_channels: int=3,
                 d_model: int=1152,
                 num_heads: int=12,
                 n_layers: int=27,
                 hidden_mlp: int=4303,
                 image_size: int=224,
                 patch_size: int=14,
                 eps: float=1e-06):
        super().__init__()
        """
        (vision_tower): SiglipVisionModel(
            (vision_model): SiglipVisionTransformer(
                (embeddings): SiglipVisionEmbeddings(
                    (...)
                )
                (encoder): SiglipEncoder(
                    (layers): ModuleList(
                        (0-26): 27 x SiglipEncoderLayer(
                            (self_attn): SiglipAttention(
                                (...)
                            )
                        )
                    )
                )
                (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
            )
        )
        """
        self.vision_model = SiglipVisionTransformer(
            in_channels=in_channels,
            d_model = d_model,
            num_heads = num_heads,
            n_layers = n_layers,
            hidden_mlp = hidden_mlp,
            image_size = image_size,
            patch_size = patch_size,
            eps = eps
        )

    def forward(self, input: torch.Tensor):
        # [B, C, H, W] => [B, num_patches, d_model]
        return self.vision_model(input)