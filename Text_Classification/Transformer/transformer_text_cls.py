import numpy as np
import pandas as pd
import torch
import re
import string
import torch.nn as nn
import torchtext 
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score
)
from torch.utils.data import DataLoader
from datasets import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import softmax
torchtext.disable_torchtext_deprecation_warning()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Config: 
    sequence_length: int = 128
    vocab_size: int = 10000
    embedding_dim: int = 256
    ff_dim: int = 128
    num_classes: int = 2
    num_layer: int = 2
    num_heads: int = 4
    dropout_prob: float = 0.1
 
    batch_size: int = 32
    num_epochs: int = 25
    lr: float = 0.001

    path_model: str = "../Transformer/checkpoint/transformer_tex_cls.pth"
    path_result: str = "../Transformer/results/transformer_text_cls_{mode}.png"


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 vocab_size: int, 
                 max_length: int, 
                 device: str):
        super().__init__()
        self.device = device
        self.embed_model = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim
        )

        self.pos_embed = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(device=self.device)
        token_embed = self.embed_model(x)
        position_embed = self.pos_embed(positions)
        return token_embed + position_embed


class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_dim: int,
                 dropout_prob: float = 0.1):
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads, # create num_heads attention
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ff_dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=ff_dim, out_features=embed_dim)
        )

        self.layernorm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
    
    def forward(self, query, key, value): 
        # query, key, value: [N, seq_len, embed_dim]
        attn_output, attn_output_weights = self.attn(query, key, value)
        # print("attn_output: ", attn_output.size()) # => output model same input: [N, seq_len, embed_dim]
        # print("attn_output_weights: ", attn_output_weights.size()) => softmax(Q@K.T): [N, seq_len, seq_len]

        attn_output = self.dropout1(attn_output) # [N, seq_len, embed_dim]
        out_1 = self.layernorm1(query + attn_output) # [N, seq_len, embed_dim]
        ffn_output = self.ffn(out_1) # [N, seq_len, embed_dim]
        ffn_output = self.dropout2(ffn_output) # [N, seq_len, embed_dim]
        out_2 = self.layernorm2(out_1 + ffn_output) # [N seq_len, embed_dim]
        return out_2 # [N,seq_len, embedim]


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_heads: int, 
                 num_layers: int,
                 embed_dim: int,
                 sequence_length: int,
                 vocab_size: int,
                 ff_dim: int,
                 dropout_prob: float, 
                 device):
        super().__init__()
        self.custom_embed = TokenAndPositionEmbedding(embed_dim=embed_dim, 
                                                      vocab_size=vocab_size,
                                                      max_length=sequence_length,
                                                      device=device)
        # create num_layers layer encoder
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim, 
                              num_heads=num_heads,
                              ff_dim=ff_dim,
                              dropout_prob=dropout_prob) 
            for i in range(num_layers)]
        )

    
    def forward(self, x):
        embedding = self.custom_embed(x) # [N, seq_len, embed_dim]
        for layer in self.encoder_layers:
            output = layer(embedding, embedding, embedding)
        return output # [N, seq_len, embed_dim]


class TransformerClassifier(nn.Module):
    def __init__(self, 
                 num_heads: int, 
                 num_layers: int,
                 embed_dim: int,
                 sequence_length: int,
                 vocab_size: int,
                 ff_dim: int,
                 dropout_prob: float, 
                 num_classes: int,
                 device):
        super().__init__()
        self.encoder_layer = TransformerEncoder(
            num_heads=num_heads,
            num_layers=num_layers,
            embed_dim=embed_dim,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            ff_dim=ff_dim,
            dropout_prob=dropout_prob,
            device=device
        )

        self.fc1 = nn.Linear(in_features=embed_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.polling = nn.AvgPool1d(kernel_size=sequence_length)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.encoder_layer(x) # [N, seq_len, embed_dim]
        output = self.polling(output.permute(0, 2, 1)).squeeze() # [N, seq_len]
        output = self.dropout(output) # [N, seq_len]
        output = self.fc1(output) # [N, 128]
        output = self.relu(output) # [N, 128]
        output = self.fc2(output) # [N, num_classes]
        return output
    

class ProcessingData:

    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
        self.train_data, self.val_data, self.test_data = self._prepare_data(df_train, df_val, df_test)
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab()
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

    def _prepare_data(self, df_train, df_val, df_test) -> tuple[Dataset, Dataset, Dataset]:
        
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
        
        return (
            Dataset.from_pandas(df_train),
            Dataset.from_pandas(df_val),  
            Dataset.from_pandas(df_test)
        )

    def _build_vocab(self):
        """build vocabulary to tokenizer text"""

        def yield_tokens(data_iter):
            for data in data_iter:
                yield self.tokenizer(data)
        
        vocab = build_vocab_from_iterator(
            iterator=yield_tokens(self.train_data['sentence']),
            min_freq=3,
            max_tokens=Config.vocab_size,
            specials=["<pad>", "</s>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab


    def _collate_batch(self, batch, 
                       seq_length: int = 500) :
        
        text_list, label_list = [], []
        for sample in batch:
            label_list.append(sample['label'])

            text_processed = self.text_pipeline(sample['sentence'])[:seq_length]
            if len(text_processed) < seq_length:
                pad_size = seq_length - len(text_processed) - 1
                text_processed = text_processed + [self.vocab['</s>']] + [self.vocab['<pad>']] * pad_size
            text_list.append(text_processed)

        input_ids = torch.tensor(text_list, dtype=torch.int64)
        label_ids = torch.tensor(label_list, dtype=torch.int64)
        return input_ids, label_ids
    

    def create_dataloader(self):
        return (
            DataLoader(dataset = self.train_data, 
                       batch_size=Config.batch_size,
                       collate_fn=self._collate_batch,
                       shuffle=True, ),
            DataLoader(dataset=self.val_data, 
                       batch_size=Config.batch_size,
                       collate_fn=self._collate_batch,
                       shuffle=False), 
            DataLoader(dataset=self.test_data,
                       batch_size=Config.batch_size,
                       collate_fn=self._collate_batch,
                       shuffle=False)
        )


def training(model: TransformerClassifier, 
             criterion: nn.CrossEntropyLoss, 
             optimizer: torch.optim.AdamW, 
             train_dataset: DataLoader, 
             val_dataset: DataLoader, 
             num_epochs: int,
             device):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for num_epoch in range(num_epochs):
        model.train()
        train_epoch_acc, train_epoch_loss, total_train = 0.0, 0.0, 0.0
        for inputs, labels in train_dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
        
        val_epoch_acc, val_epoch_loss, total_eval = 0.0, 0.0, 0.0
        
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataset:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss = criterion(output, labels)
        
    
    return train_accuracies, val_accuracies, train_losses, val_losses


def inference():
    pass


def plot_results():
    pass 

def save_checkpoint():
    pass


def load_checkpoint():
    pass


def main():
    # ----------------- DEBUG CODE 
    # model = TransformerBlock(
    #     embed_dim=256, 
    #     num_heads=1,
    #     ff_dim=128,
    #     dropout_prob=0.1,
    # )   
    # embedding_model = TokenAndPositionEmbedding(
    #     embed_dim=256, 
    #     vocab_size=20000,
    #     max_length=100,
    #     device=DEVICE
    # )
    # mock_data = torch.randint(low=0, high=2, size=(32, 100))
    # embedding = embedding_model(mock_data)
    # output = model(embedding, embedding, embedding)

    classifier = TransformerClassifier(
        num_heads=Config.num_heads,
        num_layers=Config.num_layer,
        embed_dim=Config.embedding_dim,
        sequence_length=Config.sequence_length,
        vocab_size=Config.vocab_size,
        ff_dim=Config.vocab_size,
        dropout_prob=Config.dropout_prob,
        num_classes=Config.num_classes,
        device=DEVICE
    )

    mock_data = torch.randint(low=0, high=2, size=(Config.batch_size, Config.sequence_length))
    output = classifier(mock_data)
    print(output.size())

if __name__ == "__main__":
    main()