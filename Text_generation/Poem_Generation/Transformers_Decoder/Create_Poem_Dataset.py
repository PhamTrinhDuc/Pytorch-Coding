import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Vectorization import Vocab


df = pd.read_csv("poem-datasets.csv")
vocab = Vocab(df)
corpus = vocab.vocab

vectorizer = vocab.vectorize
tokenizer = vocab.tokenizer
PAD_TOKEN = corpus['<pad>']
MAX_SEQ_LEN =  25
TRAIN_BS = 256


class PoemDataset(Dataset):
    def __init__(self, df, tokenizer, vectorizer, max_seg_len):
        self.df = df
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.max_seg_len = max_seg_len
        self.input_seqs, self.target_seqs, self.padding_masks = self.create_sample(self.df)

    
    def split_content(self, content):
        samples = []

        poem_parts = content.split("\n\n") # lấy ra từng khổ thơ

        for poem_part in poem_parts:
            poem_in_line = poem_part.split("\n") # lấy ra từng dòng thơ
            if len(poem_in_line) == 4:
                samples.append(poem_in_line)
        
        return samples
    

    def create_padding(self, input_ids, pad_token_id=PAD_TOKEN):
        return [0 if token_id==pad_token_id else 1 for token_id in input_ids]
    

    def prepare_sample(self, sample): # sample: 1 khổ thơ
        input_seqs = []
        target_seqs = []
        padding_masks = []

        input_text = '<sos> ' + '<eol>'.join(sample) + ' <eol> <eos>'
        input_ids = self.tokenizer(input_text)
        # print(input_ids)

        for idx in range(1, len(input_ids)):
            input_seq = ' '.join(input_ids[:idx])
            target_seq = ' '.join(input_ids[1: idx+1])

            input_seq = self.vectorizer(input_seq)
            target_seq = self.vectorizer(target_seq)
            padding_mask = self.create_padding(input_seq)

            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
            padding_masks.append(padding_mask)

        return input_seqs, target_seqs, padding_masks
    


    def create_sample(self):

        input_seqs = []
        target_seqs = []
        padding_masks = []

        for idx, row in self.df.iterrows():
            samples = self.split_content(row['content'])

            for sample in samples:
                sample_input_seqs, sample_target_seqs, sample_padding_masks = self.prepare_sample(sample=sample)


                input_seqs += sample_input_seqs
                target_seqs += sample_target_seqs
                padding_masks += sample_padding_masks
        
        input_seqs = torch.tensor(input_seqs, dtype=torch.long)
        target_seqs = torch.tensor(target_seqs, dtype=torch.long)
        padding_masks = torch.tensor(padding_masks, dtype=torch.float)

        return input_seqs, target_seqs, padding_masks
    

    def __len__(self):
        return len(self.input_seqs)
    
    def __getitem__(self, idx):
        input_seqs = self.input_seqs[idx]
        target_seqs = self.target_seqs[idx]
        padding_masks = self.padding_masks[idx]

        return input_seqs, target_seqs, padding_masks
    

def create_dataloader():
    dataset = PoemDataset(
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        df=df,
        max_seg_len=MAX_SEQ_LEN
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=TRAIN_BS,
        shuffle=False
    )

    return data_loader

if __name__ == "__main__":

    data_loader= create_dataloader()
    input_seqs, target_seqs, padding_masks = next(iter(data_loader))
    print(input_seqs[0])
    print(target_seqs[0])
    print(padding_masks[0])
    print(vocab.decode(input_seqs[0]))


