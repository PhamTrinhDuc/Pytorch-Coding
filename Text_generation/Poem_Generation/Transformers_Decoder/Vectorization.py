import pandas as pd
from torchtext.vocab import build_vocab_from_iterator



DATASET_PATH = "poem-datasets.csv"
MAX_SEQ_LEN = 25
df = pd.read_csv(DATASET_PATH)


class Vocab:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df['content'] = self.df['content'].apply(lambda x: self.normal_text(x))
        self.vocab = self.build_vocab()


    def normal_text(self, text: str):
        text = text.strip()
        return text
    

    def tokenizer(self, text: str):
        return text.split()
    

    def yield_tokenizer(self):
        for idx, row in self.df.iterrows():
            yield self.tokenizer(row['content'])
    

    def build_vocab(self):
        vocab = build_vocab_from_iterator(
            self.yield_tokenizer(),
            specials=['<unk>', '<pad>', '<sos>', '<eos>', '<eol>']
        )

        vocab.set_default_index(vocab['<unk>'])
        return vocab
    

    def pad_and_truncate(self, input_ids: list[int]):
        PAD_TOKEN = self.vocab['<pad>']
        if len(input_ids) > MAX_SEQ_LEN:
            input_ids = input_ids[:MAX_SEQ_LEN]
        else:
            input_ids += [PAD_TOKEN] * (MAX_SEQ_LEN - len(input_ids))

        return input_ids


    def vectorize(self, text):
        input_ids = [self.vocab[token] for token in self.tokenizer(text)]
        input_ids = self.pad_and_truncate(input_ids)

        return input_ids
    

    def decode(self, input_ids):
        return [self.vocab.get_itos()[token_id] for token_id in input_ids]
    

if __name__ == "__main__":
    vocab = Vocab(df)
    text = df.loc[1]['content'].split("\n\n")[0].split("\n")[0]
    text_vectorize = vocab.vectorize(text)
    print(text_vectorize)