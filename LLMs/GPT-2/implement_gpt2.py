import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass


@dataclass
class GPTConfig124M:
    vocab_size: int = 50257
    context_len: int = 1024
    embed_dim:int =  768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: 0.05
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
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        pass

class FeedForward(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        pass

class TransformerBlock(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        pass

class GPTModel(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        pass

def main():
    pass

if __name__ == "__main__":
    main()