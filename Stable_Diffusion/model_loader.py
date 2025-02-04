from clip import CLIP
from vae.encoder import VaeEncoder
from vae.decoder import VaeDecoder
from diffusion import Diffusion
from config import VaeArgs, CLIPArgs, UnetArgs

def preload_models_from_standard_weights():
    
    encoder = VaeEncoder(in_encode=VaeArgs.in_channels,
                         hidden_encode=VaeArgs.hidden_encode_channels,
                         out_encode=VaeArgs.out_channels, 
                         num_groups=VaeArgs.num_groups)
    
    decoder = VaeDecoder(in_decode=VaeArgs.out_channels,
                         out_decode=VaeArgs.in_channels,
                         hidden_decode=VaeArgs.hidden_decode_channels,
                         num_groups=VaeArgs.num_groups)
    clip = CLIP(vocab_size=CLIPArgs.vocab_size,
                seq_len=CLIPArgs.seq_len,
                d_model=CLIPArgs.d_model,
                num_layers=CLIPArgs.num_layers,
                num_heads=CLIPArgs.num_heads)
    
    diffusion = Diffusion(config=UnetArgs)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion
    }
    