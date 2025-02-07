import torch
import torch.nn as nn
from siglip_modeling import SiglipVisionModel

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, 
                 head_dim: int,
                 base: int=1000):
        super().__init__()

        inv_freq = 1.0/(base**(torch.arange(0, head_dim, 2, dtype=torch.int64).float()/head_dim))
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistent=False)

    def forward(self, 
                x: torch.Tensor, # Q, K
                position_ids: torch.Tensor):
        
        # shape x: [B, num_heads, seq_len, head_dim]
        device, device_type = x.device, x.device.type
        # [head_dim/2, ]
        self.inv_freq.to(device=device)
        # [head_dim/2, ] => [1, head_dim/2, 1] => [B, head_dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(x.shape[0], -1, 1)
        # [B, seq_len] => [B, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):

            # [B, head_dim/2, 1] @ [B, 1, seq_len] = [B, head_dim/2, seq_len]
            # caculate theta: theta = position / 100000 ** (2*k//d)
            freqs =torch.matmul(inv_freq_expanded, position_ids_expanded) 
            # [B, head_dim/2, seq_len] => [B, seq_len, head_dim/2]
            freqs = freqs.transpose(1, 2)
            # [B, seq_len, head_dim/2] => [B, seq_len, head_dim]
            freqs = torch.cat(tensors=(freqs, freqs), dim=-1)
            # caculate cos, sin theta
            cos_freqs = freqs.cos()
            sin_freqs = freqs.sin()
        return cos_freqs, sin_freqs


def rotate_half(x: torch.Tensor):
    """
    x[..., : x.shape[-1] // 2]: 
        lấy tất cả các phần tử của các chiều trước đó và chỉ lấy nửa đầu của chiều cuối cùng.
    
    torch.cat((-x2, x1), dim=-1): 
        nối tensor -x2 và x1 theo chiều cuối cùng (dim=-1), tạo nên tensor mới với thứ tự các cặp phần tử được "quay".
    """
    # x: [B, num_heads, seq_len, head_dim]

    # [..., head_dim] => [..., head_dim//2]
    x1 = x[..., : x.shape[-1]//2]
    # [..., head_dim] => [..., head_dim//2]
    x2 = x[..., x.shape[-1]//2: ]

    return torch.cat(tensors=(-x2, x1), dim=-1)

def apply_rotary_pos_embed(Q: torch.Tensor, K: torch.Tensor, 
                           cos: torch.Tensor, sin: torch.Tensor, 
                           head_dim_dimension: int=1):
    
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # [B, seq_len, head_dim] => [B, 1, seq_len, head_dim]
    cos = cos.unsqueeze(dim=head_dim_dimension) # add head_dim dimension
    sin = sin.unsqueeze(dim=head_dim_dimension) # add head_dim dimension

    Q_embed = Q*cos + rotate_half(Q)*sin
    K_embed = K*cos + rotate_half(K)*sin

    return Q_embed, K_embed

########################################
########                       #########
########################################

class KVCache:
    def __init__(self):
        # List of key tensors. Each element of this list corresponds to a layer.
        self.key_cache: list[torch.Tensor] = []
        # List of value tensors. Each element of this list corresponds to a layer.
        self.value_cache: list[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # a item key_cache has shape: [B, num_heads, seq_len, head_dim]
            return self.key_cache[0].shape[-2] # get num tokens saved in cache
    
    def update(self, 
               key_state: torch.Tensor,
               value_state: torch.Tensor, 
               layer_idx: int):
        
        # If there is no key and value cache for this layer, 
        # create a new one by adding the element to the list.
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_state)
            self.value_cache.append(value_state)
        
        else:
            # If there is a cache, concatenate the new tensors with the existing tensors of the specified layer_idx
            # Each tensor has dimensions: [B, num_heads, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat(tensors=(self.key_cache[layer_idx], key_state), dim=2)
            self.value_cache[layer_idx] = torch.cat(tensors=(self.value_cache[layer_idx], value_state), dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    

def repeat_kv(kv_tensor: torch.Tensor, n_repeat: int):
    if n_repeat == 1:
        return kv_tensor
    
    B, num_head_kv, seq_len, head_dim = kv_tensor.size()
    # [B, num_head_kv, seq_len, head_dim] => [B, num_head_kv, 1, seq_len, head_dim]
    kv_tensor = kv_tensor[..., None, :, :]
    # [B, num_head_kv, 1, seq_len, head_dim] => [B, num_head_kv, n_repeat, seq_len, head_dim]
    kv_tensor = kv_tensor.expand(B, num_head_kv, n_repeat, seq_len, head_dim)
    # [B, num_head_kv, n_repeat, seq_len, head_dim] => [B, num_head_kv*n_repeat, seq_len, head_dim]
    return kv_tensor.reshape(B, num_head_kv*n_repeat, seq_len, head_dim)

########################################
########                       #########
########################################

class GemmaSdpaAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads_q: int,
                 num_heads_kv: int, 
                 base: int=10000,
                 is_bias: bool=False):
        super().__init__()
        """
        (self_attn): GemmaSdpaAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=256, bias=False)
            (v_proj): Linear(in_features=2048, out_features=256, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary_emb): GemmaRotaryEmbedding()
          )
        """
        assert d_model % num_heads_q == 0, "d_model must be divible num_heads_q"

        self.head_dim = d_model // num_heads_q
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv # parameter for grouped multi query attention
        self.num_kv_group = num_heads_q // num_heads_kv

        self.q_proj = nn.Linear(in_features=d_model,
                                out_features=num_heads_q*self.head_dim,
                                bias=is_bias)
        self.k_proj = nn.Linear(in_features=d_model,
                                out_features=num_heads_kv*self.head_dim,
                                bias=is_bias)
        self.v_proj = nn.Linear(in_features=d_model,
                                out_features=num_heads_kv*self.head_dim,
                                bias=is_bias)
        self.o_proj = nn.Linear(in_features=d_model,
                                out_features=d_model,
                                bias=is_bias)
        self.rotary_embeder = GemmaRotaryEmbedding(head_dim=self.head_dim, base=base)

    def split_heads(self, x: torch.Tensor, num_heads: int):
        batch_size, seq_len, d_model = x.size()
        # [N, seq_len, num_heads(q or kv) * head_dim] -> [N, seq_len, num_heads(q or kv), head_dim] 
        out = x.reshape(batch_size, num_heads, seq_len, self.head_dim)
        return out
    
    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     is_mask: bool=False):
        # print("shape Q: ", Q.shape)
        # print("shape K: ", K.shape)
        seq_len_q = Q.size(dim=2) # seq_len input
        seq_len_kvcache = K.size(2) # seq_len of K and V after upadated by KVCache
        # [..., seq_len_q, head_dim]@[..., head_dim, seq_len_kv_cache] = [.., seq_len_q, seq_len_kv_cache]
        matmul_QK = torch.matmul(Q, K.transpose(-2, -1))
        matmul_QK = matmul_QK * self.num_heads_q**-0.5

        if is_mask:
            mask = torch.triu(torch.ones(seq_len_q, seq_len_kvcache), diagonal=1).bool()[:seq_len_q, :seq_len_kvcache]
            matmul_QK.masked_fill_(mask=mask, value=-torch.inf)
        
        attn_weight = nn.functional.softmax(matmul_QK, dim=-1)
        # [.., seq_len_q, seq_len_kv_cache]@[.., seq_len_kv_cache, head_dim]=[..., seq_len_q, head_dim]
        attn_output = torch.matmul(attn_weight, V)
        return attn_output, attn_weight


    def forward(self, 
                x: torch.Tensor, 
                position_ids: torch.LongTensor, 
                kv_cache: KVCache, 
                layer_idx: int,
                is_attn_mask: bool=False,):
        # shape x: [B, seq_len, d_model]

        # [B, seq_len, d_model]=>[B, seq_len, num_heads_q*head_dim]=>[B, num_heads_q, seq_len, head_dim] 
        Q = self.split_heads(self.q_proj(x), num_heads=self.num_heads_q)
        # [B, seq_len, d_model]=>[B, seq_len, num_heads_kv*head_dim]=>[B, num_heads_kv, seq_len, head_dim]
        K = self.split_heads(self.k_proj(x), num_heads=self.num_heads_kv)
        # [B, seq_len, d_model] => [B, seq_len, num_heads_kv*head_dim]=>[B, num_heads_kv, seq_len, head_dim]
        V = self.split_heads(self.v_proj(x), num_heads=self.num_heads_kv)

        # [B, num_heads_kv, seq_len, head_dim], [B, num_heads_kv, seq_len, head_dim]
        cos, sin = self.rotary_embeder(V, position_ids)
        # [B, num_heads_q, seq_len, head_dim], [B, num_heads_kv, seq_len, head_dim]
        Q, K = apply_rotary_pos_embed(Q=Q, K=K, cos=cos, sin=sin)

        if kv_cache is not None:
            K_cache, V_cache = kv_cache.update(key_state=K, value_state=V, layer_idx=layer_idx)
        
        # Repeat the key and values to match the number of heads of the query
        # [B, num_heads_kv, seq_len, head_dim] => [B, num_heads_q, seq_len, head_dim]
        K = repeat_kv(kv_tensor=K_cache, n_repeat=self.num_kv_group)
        V = repeat_kv(kv_tensor=V_cache, n_repeat=self.num_kv_group)
        # [B, num_heads_q, seq_len, head_dim]
        attn_output, attn_weight = self.scaled_dot_product_attention(Q=Q, K=K, V=V, is_mask=is_attn_mask)
        # [B, num_heads_q, seq_len, head_dim] => [B, seq_len, num_heads_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B, seq_len, num_heads_q, head_dim] => [B, seq_len, d_model]
        attn_output = attn_output.view(*x.size())
        # [B, seq_len, d_model] =>  [B, seq_len, d_model]
        attn_output = self.o_proj(attn_output)
        return attn_output


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
    

class GemmaMLP(nn.Module):
    def __init__(self, 
                 d_model: int,
                 hidden_mlp: int,
                 is_bias: bool=False):
        super().__init__()
        self.gate_proj = nn.Linear(in_features=d_model, out_features=hidden_mlp, bias=is_bias)
        self.up_proj = nn.Linear(in_features=d_model, out_features=hidden_mlp, bias=is_bias)
        self.down_proj = nn.Linear(in_features=hidden_mlp, out_features=d_model, bias=is_bias)
        self.act_fn = PytorchGELUTanh()

    def forward(self, x: torch.Tensor):
        # [B, seq_len, d_model] => [B, seq_len, hidden_mlp]
        output = self.gate_proj(x)
        # [B, seq_len, hidden_mlp] => [B, seq_len, hidden_mlp]
        output = self.act_fn(output)
        # [B, seq_len, hidden_mlp] * [B, seq_len, hidden_mlp] = [B, seq_len, hidden_mlp]
        output = self.down_proj(output * self.up_proj(x))
        return output # [B, seq_len, hidden_mlp]


class GemmaRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(self.d_model))
    
    def forward(self, x):
        x = x.float() # [N, seq_len, d_model]
        norm = torch.sqrt(torch.mean(input=x**2, dim=-1, keepdim=True)) + self.eps # [N, seq_len, d_model]
        x_hat = x / norm  # [N, seq_len, d_model]
        y = x_hat * self.weights # [N, seq_len, d_model]
        return y 


class GemmaDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads_q: int,
                 num_heads_kv: int,
                 hidden_mlp: int,
                 base: int=10000,
                 eps_norm: float=1e-06,
                 is_bias=False):
        super().__init__()
        """
        GemmaDecoderLayer(
          (self_attn): GemmaSdpaAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=256, bias=False)
            (v_proj): Linear(in_features=2048, out_features=256, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary_emb): GemmaRotaryEmbedding()
          )
          (mlp): GemmaMLP(
            (gate_proj): Linear(in_features=2048, out_features=16384, bias=False)
            (up_proj): Linear(in_features=2048, out_features=16384, bias=False)
            (down_proj): Linear(in_features=16384, out_features=2048, bias=False)
            (act_fn): PytorchGELUTanh()
          )
          (input_layernorm): GemmaRMSNorm()
          (post_attention_layernorm): GemmaRMSNorm()
        )
        """
        self.attn = GemmaSdpaAttention(d_model=d_model, 
                                        num_heads_q=num_heads_q, 
                                        num_heads_kv=num_heads_kv, 
                                        base=base, 
                                        is_bias=is_bias)

        self.mlp = GemmaMLP(d_model=d_model, 
                            hidden_mlp=hidden_mlp, 
                            is_bias=is_bias)
        self.input_layernorm = GemmaRMSNorm(d_model=d_model, eps=eps_norm)
        self.post_attention_layernorm= GemmaRMSNorm(d_model=d_model, eps=eps_norm)

    def forward(self,
                x: torch.Tensor,
                position_ids: torch.LongTensor, 
                kv_cache: KVCache,
                layer_idx: int,
                is_attn_mask: bool=False):
        
        residual = x
        # [B, seq_len, d_model]
        output = self.input_layernorm(x)
        # [B, seq_len, d_model]
        output = self.attn(x, position_ids, kv_cache, layer_idx, is_attn_mask)
        # [B, seq_len, d_model] + [B, seq_len, d_model] = [B, seq_len, d_model]
        output = output + residual
        # [B, seq_len, d_model]
        residual = output
        # [B, seq_len, d_model]
        output = self.post_attention_layernorm(output)
        # [B, seq_len, d_model]
        output = self.mlp(output)
        # [B, seq_len, d_model]
        output = output + residual
        return output # [B, seq_len, d_model]
    

########################################
########                       #########
########################################

class GemmaModel(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 vocab_size: int,
                 d_model: int,
                 num_heads_q: int,
                 num_heads_kv: int,
                 hidden_mlp: int,
                 base: int=10000,
                 eps_norm: float=1e-06,
                 is_bias=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=vocab_size, 
                                         embedding_dim=d_model)
        self.layers = nn.ModuleList(modules=[
            GemmaDecoderLayer(
                d_model=d_model,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                hidden_mlp=hidden_mlp,
                base=base,
                eps_norm=eps_norm,
                is_bias=is_bias
            )
            for _ in range(num_layers)
        ])
        self.norm = GemmaRMSNorm(d_model=d_model, eps=eps_norm)

    def forward(self, 
                tokens_ids: torch.Tensor, 
                position_ids: torch.LongTensor,
                kv_cache: KVCache,
                is_attn_mask: bool=False):
        # [B, seq_len] => [B, seq_len, d_model]
        embedding = self.embed_tokens(tokens_ids)
        for layer_idx, layer in enumerate(self.layers):
            # [B, seq_len, d_model]
            output = layer(embedding, position_ids, kv_cache, layer_idx, is_attn_mask)
        # [B, seq_len, d_model]
        output = self.norm(output)
        return output

########################################
########                       #########
########################################

class GemmaForCausalLM(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 vocab_size: int,
                 d_model: int,
                 num_heads_q: int,
                 num_heads_kv: int,
                 hidden_mlp: int,
                 base: int=10000,
                 eps_norm: float=1e-06,
                 is_bias=False):
        super().__init__()
        self.model = GemmaModel(
            num_layers=num_layers,
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            hidden_mlp=hidden_mlp,
            base=base,
            eps_norm=eps_norm,
            is_bias=is_bias
        )
        self.lm_head = nn.Linear(in_features=d_model, out_features=vocab_size, bias=is_bias)

    def forward(self, 
                token_ids: torch.LongTensor,
                position_ids: torch.LongTensor,
                kv_cache: KVCache,
                is_attn_mask: bool=False):
        # [B, seq_len] => [B, seq_len, d_model]
        output = self.model(token_ids, position_ids, kv_cache, is_attn_mask)
        # [B, seq_len, d_model] => [B, seq_len, vocab_size]
        logits = self.lm_head(output)

        return_data = {
            "logits": logits
        }
        if kv_cache is not None:
            # Return the updated cache
            return_data['kv_cache'] = kv_cache
        return return_data
        

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 padding_idx: int=0):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features,
                                bias=True)
    
    def forward(self, x: torch.Tensor):
        # [B, seq_len, d_model_vision] => [B, seq_len, d_model_text]
        return self.linear(x)
    

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config_text, config_vision):
        super().__init__()
        self.vision_tower = SiglipVisionModel(
            config_vision
        )

        self.mlti_modal_projection = PaliGemmaMultiModalProjector(
            in_features=config_vision.d_model,
            out_features=config_text.d_model
        )

        self.language_model = GemmaForCausalLM(config_text)

    def forward(self, 
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                position_ids: torch.LongTensor,
                is_attn_mask: bool=False):
        pass