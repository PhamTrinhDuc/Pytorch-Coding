import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import re
import string
import torch.nn as nn
import torchinfo
import torchtext 
from torch.utils.data import DataLoader
from datasets import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
torchtext.disable_torchtext_deprecation_warning()



class RNNClsText(nn.Module):
    def __init__(self, embed_dim: str, 
                 hidden_dim: str, 
                 vocab_size: int,
                 output_dim: int = 2, 
                 num_layer: int = 2):
        super().__init__()
        self.embed_model = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim)
        
        self.rnn = nn.RNN(input_size=embed_dim, 
                                hidden_size=hidden_dim, 
                                num_layers=num_layer, 
                                bidirectional=True, 
                                batch_first=True)
        
        self.fully_connected = nn.Linear(in_features=hidden_dim, 
                                         out_features=output_dim)

    def forward(self, x):
        embedding = self.embed_model(x)
        rnn_output, hidden_output = self.rnn(embedding) # input: [N, sequence_length, embed_dim]
        last_hidden = hidden_output[-1, :, :] # hidden_output: [num_layers, batch_size, hidden_dim]
        # last_hidden = rnn_output[:, -1, :] # rnn_output: [N, sequence_length, hidden_dim]
        output = self.fully_connected(last_hidden)
        return last_hidden


class ProcessingData:
    vocab_size = 20000
    batch_size = 64

    def __init__(self):
        self.train_data, self.val_data, self.test_data = self._prepare_data()
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab()
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

    def _prepare_data(self) -> tuple[Dataset, Dataset, Dataset]:
        df_train = pd.read_csv("RNN/data/cls_text/train.csv")
        df_val = pd.read_csv("RNN/data/cls_text/val.csv")
        df_test = pd.read_csv("RNN/data/cls_text/test.csv")

        def preprocess_text(text) -> str:
            # remove URLs https://www.
            url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
            text = url_pattern.sub(r" ", text)

            # remove HTML Tags: <>
            html_pattern = re.compile(r'<[^<>]+>')
            text = html_pattern.sub(" ", text)

            # remove puncs and digits
            replace_chars = list(string.punctuation + string.digits)
            for char in replace_chars:
                text = text.replace(char, " ")

            # remove emoji
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U0001F1F2-\U0001F1F4"  # Macau flag
                u"\U0001F1E6-\U0001F1FF"  # flags
                u"\U0001F600-\U0001F64F"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U0001F1F2"
                u"\U0001F1F4"
                u"\U0001F620"
                u"\u200d"
                u"\u2640-\u2642"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r" ", text)

            # normalize whitespace
            text = " ".join(text.split())

            # lowercasing
            text = text.lower()
            return text

        df_train['sentence'].apply(preprocess_text)
        df_val['sentence'].apply(preprocess_text)
        df_test['sentence'].apply(preprocess_text)
        
        return Dataset.from_pandas(df_train), \
            Dataset.from_pandas(df_val), Dataset.from_pandas(df_test)


    def _build_vocab(self):
        """build vocabulary to tokenizer text"""
        vocab_size = self.vocab_size

        def yield_tokens(data_iter):
            for data in data_iter:
                yield self.tokenizer(data)
        
        vocab = build_vocab_from_iterator(
            iterator=yield_tokens(self.train_data['sentence']),
            min_freq=3,
            max_tokens=vocab_size,
            specials=["<pad>", "</s>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def _collate_batch(self, batch, 
                       seq_length: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
        
        text_list, label_list = [], []
        for sample in batch:
            label_list.append(sample['label'])

            text_processed = self.text_pipeline(sample['text'])[:seq_length]
            if len(text_processed) < seq_length:
                pad_size = seq_length - len(text_processed)
                text_processed = text_processed + [self.vocab['</s>']] + self.vocab['<pad>'] * pad_size
            text_list.append(text_processed)

        input_ids = torch.tensor(text_list, dtype=torch.int64)
        label_ids = torch.tensor(label_list, dtype=torch.int64)
        return input_ids, label_ids
    
    def create_dataloader(self):
        return DataLoader(dataset = self.train_data, 
                          batch_size=self.batch_size,
                          collate_fn=self._collate_batch,
                          shuffle=True, ), \
               DataLoader(dataset=self.val_data, 
                          batch_size=self.batch_size,
                          collate_fn=self._collate_batch,
                          shuffle=False)

def main():
    processor = ProcessingData()
    # print(processor.vocab.get_stoi())
    
    # ----------------- init data loader
    train_dataloader, val_dataloader = processor.create_dataloader()

    # ---------------- init model
    model = RNNClsText(embed_dim=128, 
                       hidden_dim=64,
                       vocab_size=processor.vocab_size,
                       num_layer=2)
    
    torchinfo.summary(model=model)


if __name__ == "__main__":
    main()