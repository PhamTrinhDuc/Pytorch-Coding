import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


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
        drop_last=is_drop_last, 
    )
    return dataloader


def text_to_tokens_ids(text: str, tokenizer) -> torch.Tensor:
    tokens_ids = tokenizer.encode(text).unsqueeze(0) # add batch dimension
    encoded_tensor = torch.tensor(data=tokens_ids, dtype=torch.long)
    return encoded_tensor

def token_ids_to_text(tokens_ids, tokenizer) -> str:
    flat = tokens_ids.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text

