import torch
from dataclasses import dataclass

@dataclass
class ConfigDataset:
    data_path: str = "data/styles.csv"
    test_size: float = 0.1
    target_size: tuple[int, int] = (80, 80)
    batch_size: int = 32


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


@dataclass
class TextEncoderConfig:
    vocab_size: int = 50257 # vocab of tokenizer gpt2
    d_model: int = 64
    ff_dim: int = d_model * 4
    max_seq_len: int = 33 # 32 + 1 token CLS
    embedding_image: int = 128
    n_heads: int = 8
    n_layers: int = 4
    drop_rate: float = 0.05


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 50
    path_model: str = "./checkpoint.pth"
    device: str = 'cuda' if torch.cuda.is_available() else "cpu"
