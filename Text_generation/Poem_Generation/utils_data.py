import pandas as pd
import torch
from dataclasses import dataclass
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from typing import List, Callable, Tuple
from torch.utils.data import Dataset, DataLoader

tokenizer = get_tokenizer("basic_english")

class VocabularyBuilder:
    def __init__(self, df: pd.DataFrame, max_seq_len: int):
        self.df = df
        self.max_seq_len = max_seq_len
        self.df['content'] = self.df['content'].apply(self._normalize_text)
        self.vocab = self._build_vocab()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.strip()

    def _yield_tokens(self):
        for _, row in self.df.iterrows():
            yield tokenizer(row['content'])

    def _build_vocab(self):
        vocab = build_vocab_from_iterator(
            iterator=self._yield_tokens(),
            specials=['<unk>', '<pad>', '<sos>', '<eos>', '<eol>']
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def pad_and_truncate(self, input_ids: List[int]) -> List[int]:
        pad_token = self.vocab['<pad>']
        if len(input_ids) < self.max_seq_len:
            return input_ids[:self.max_seq_len] + [pad_token] * (self.max_seq_len - len(input_ids))
        return input_ids[:self.max_seq_len]

    def vectorize(self, text: str) -> List[int]:
        input_ids = [self.vocab[token] for token in tokenizer(text)]
        return self.pad_and_truncate(input_ids)

    def decode(self, input_ids: List[int]) -> List[str]:
        return [self.vocab.get_itos()[token_id] for token_id in input_ids]
    

class PoemDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len: int):
        self.df = df
        self.vocab_builder = VocabularyBuilder(df=df, max_seq_len=max_seq_len)
        self.input_seqs, self.target_seqs, self.padding_masks = self._create_samples()

    def _split_content(self, content: str) -> List[List[str]]:
        return [part.split("\n") for part in content.split("\n\n") 
                if len(part.split("\n")) == 4]

    def _create_padding(self, input_ids: List[int], pad_token_id: int) -> List[int]:
        return [0 if token_id == pad_token_id else 1 for token_id in input_ids]

    def _prepare_sample(self, sample: List[str]) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        input_seqs, target_seqs, padding_masks = [], [], []
        
        input_text = f"<sos> {'<eol>'.join(sample)} <eol> <eos>"
        input_ids = tokenizer(input_text)

        for idx in range(1, len(input_ids)):
            input_seq = self.vocab_builder.vectorize(' '.join(input_ids[:idx]))
            target_seq = self.vocab_builder.vectorize(' '.join(input_ids[1:idx + 1]))
            padding_mask = self._create_padding(
                input_seq, 
                pad_token_id=self.vocab_builder.vocab['<pad>']
            )

            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
            padding_masks.append(padding_mask)

        return input_seqs, target_seqs, padding_masks

    def _create_samples(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_seqs, target_seqs, padding_masks = [], [], []

        for _, row in self.df.iterrows():
            samples = self._split_content(row['content'])
            for sample in samples:
                sample_input_seqs, sample_target_seqs, sample_padding_masks = self._prepare_sample(sample)

                input_seqs.extend(sample_input_seqs)
                target_seqs.extend(sample_target_seqs)
                padding_masks.extend(sample_padding_masks)

        return (
            torch.tensor(input_seqs, dtype=torch.long),
            torch.tensor(target_seqs, dtype=torch.long),
            torch.tensor(padding_masks, dtype=torch.float),
        )

    def __len__(self) -> int:
        return len(self.input_seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_seqs[idx], self.target_seqs[idx], self.padding_masks[idx]


def main():
    df = pd.read_csv("Text_generation/Poem_Generation/poem-datasets.csv")

    tokenizer = get_tokenizer("basic_english")
    dataset = PoemDataset(df=df, max_seq_len=25)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2
    )
    print(len(data_loader))
    for input_seqs, target_seqs, padding_masks in data_loader:
        print(input_seqs[0].shape)
        print(target_seqs[0].shape)
        print(padding_masks[0].shape)
        break

if __name__ == "__main__":
    main()