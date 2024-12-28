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
    sequence_length: int = 500
    vocab_size: int = 20000
    embedding_dim: int = 128
    hidden_dim: int = 64
    output_dim: int = 2
    num_layer: int = 2
    is_bidirectional: bool = True
    dropout_prob: float = 0.1

    batch_size: int = 64
    num_epochs: int = 25
    lr: float = 0.001

    path_model: str = "../LSTM/checkpoint/lstm_tex_cls.pth"
    path_result: str = "../LSTM/results/lstm_text_cls_{mode}.png"


class SentimentClassifier(nn.Module):
    def __init__(self, 
                 embed_dim: str, 
                 hidden_dim: str, 
                 vocab_size: int,
                 output_dim: int = 2, 
                 num_layer: int = 2,
                 dropout_prob: float = 0.1,
                 is_bidirectional: bool = False):
        super().__init__()
        self.embed_model = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.rnn = nn.LSTM(input_size=embed_dim, 
                           hidden_size=hidden_dim, 
                           num_layers=num_layer, 
                           bidirectional=is_bidirectional, 
                           batch_first=True)
        
        self.fc1 = nn.Linear(in_features=hidden_dim, 
                             out_features=16)
        
        self.fc2 = nn.Linear(in_features=16, 
                             out_features=output_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        embedding = self.embed_model(x)
        output_lstm, (hidden_lstm, cell_lstm) = self.rnn(embedding) # input: [N, sequence_length, embed_dim]
        last_hidden = hidden_lstm[-1, :, :] # hidden_lstm: [num_layers, batch_size, hidden_dim]
        # last_hidden = output_lstm[:, -1, :] # output_lstm: [N, sequence_length, hidden_dim]
        output = self.norm(last_hidden)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


class ProcessingData:

    def __init__(self):
        self.train_data, self.val_data, self.test_data = self._prepare_data()
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab()
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

    def _prepare_data(self) -> tuple[Dataset, Dataset, Dataset]:
        df_train = pd.read_csv("./data/cls_text/train.csv")
        df_val = pd.read_csv("./data/cls_text/val.csv")
        df_test = pd.read_csv("./data/cls_text/test.csv")

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


def training(model: nn.Module,
             criterion: nn.CrossEntropyLoss,
             optimizer: torch.optim.AdamW,
             train_dataloader: DataLoader,
             val_dataloader: DataLoader,
             num_epochs: int):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for num_epoch in range(num_epochs):

        running_train_loss, running_train_correct = 0.0, 0.0
        running_val_loss, running_val_correct = 0.0, 0.0
        total_train, total_val  = 0.0, 0.0
        # training model with training set
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            # Forward 
            predictions = model(inputs)

            # caculate loss behind predictions and labels
            loss = criterion(predictions, labels)
            running_train_loss += loss.item()

            # caculate accuracy
            _, predicted = torch.max(predictions, dim=1)
            total_train += labels.size(0)
            running_train_correct += (predicted==labels).sum().item()


            # backpropagation
            loss.backward()
            optimizer.step()
        train_loss = running_train_loss / len(train_dataloader)
        train_acc = 100 * running_train_correct / total_train

        # evaludate model with val set
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                # Forward
                predictions = model(inputs)
                # Caculate loss
                loss = criterion(predictions, labels)
                running_val_loss += loss.item()

                # Caculate accuracy
                _, predicted = torch.max(predictions, 1)
                running_val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = running_val_correct * 100 / total_val
        val_loss = running_val_loss / len(val_dataloader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(val_loss)
        test_accuracies.append(val_acc)

        print(f"Epoch [{num_epoch + 1}/{num_epochs}]\t Train Acc: {train_acc:.4f}\t Test Acc: {val_acc:.4f}\t Train Loss: {train_loss:.4f}\t Test Loss: {val_loss:.4f}")
    return train_accuracies, test_accuracies, train_losses, test_losses


def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: str = 'cuda',
            task_type: str = 'binary') -> dict:
    """
    Evaluate model performance with multiple metrics
    
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader containing validation/test data
        device: Device to run evaluation on ('cuda' or 'cpu')
        task_type: Type of classification task ('binary' or 'multi')
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Validate arguments
    if task_type not in ['binary', 'multi']:
        raise ValueError("task_type must be either 'binary' or 'multi'")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)

            # Get predictions and probabilities
            if task_type == 'binary':
                probs = logits[:, 1].sigmoid()
                preds = (probs > 0.5).int()
                all_probs.append(probs.cpu().numpy())
            else:
                preds = logits.argmax(dim=1)

            # Store batch results
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(
            all_labels, 
            all_preds,
            average='binary' if task_type == 'binary' else 'macro'
        ),
        'recall': recall_score(
            all_labels, 
            all_preds, 
            average='binary' if task_type == 'binary' else 'macro'
        ),
        'f1': f1_score(
            all_labels, 
            all_preds,
            average='binary' if task_type == 'binary' else 'macro'
        ),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    # Add ROC-AUC for binary classification
    if task_type == 'binary':
        all_probs = np.concatenate(all_probs)
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)

    return metrics


def inference(text: str):
    model = load_model(
        model=SentimentClassifier(
            embed_dim=Config.embedding_dim,
            hidden_dim=Config.hidden_dim,
            vocab_size=Config.vocab_size
        ), 
        path_model=Config.path_model,
        device=DEVICE
    )
    
    tokenizer = get_tokenizer("basic_english")

    vocab = ProcessingData()._build_vocab()
    text_pipeline = lambda x: vocab(tokenizer(x))

    text_list = []
    text_processed = text_pipeline(text)[:Config.sequence_length]
    if len(text_processed) < Config.sequence_length:
        pad_size = Config.sequence_length - len(text_processed)
        text_processed = text_processed + [vocab['</s>']] + [vocab['<pad>']] * pad_size
    text_list.append(text_processed)
    input_ids = torch.tensor(text_list, dtype=torch.int64)

    with torch.no_grad():
        logits = model(input_ids)
        print(logits.shape)

        probabilities = softmax(logits, dim=-1)  
        predicted_label = torch.argmax(probabilities, dim=-1).item() 
        return predicted_label


def plot_results(
        config,
        mode: str, 
        train_results: list, 
        val_results: list, 
        is_storage_results: bool = True): 
    if not isinstance(train_results, list) or not isinstance(val_results, list):
        raise ValueError("train_results and val_results must be list .")
    if len(train_results) != len(val_results):
        raise ValueError("train_results and val_results must be the same length .")
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(train_results, label=f"Train {mode}", color='blue')
    ax[0].set_title(f"Training {mode}")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("{mode}")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(val_results, label=f"Validation {mode}", color='orange')
    ax[1].set_title(f"Validation {mode}")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(f"{mode}")
    ax[1].grid(True)
    ax[1].legend()
    if is_storage_results:
        storage_results = config.path_result.format(mode=mode)
        try:
            plt.savefig(storage_results)
        except Exception as e:
            print(f"Error while store results: {e}")

    plt.show()


def save_model(model: nn.Module, path_model) -> None:
    torch.save(obj=model.state_dict(), 
               f=path_model)

def load_model(model: nn.Module, path_model: str, device) -> nn.Module:
    model.load_state_dict(torch.load(f=path_model, 
                                     map_location=torch.device(device=device)))
    return model

def main():
    processor = ProcessingData()
    # print(processor.vocab.get_stoi())
    
    # ----------------- init data loader
    train_dataloader, val_dataloader, test_dataloader = processor.create_dataloader()

    # ---------------- init model
    model = SentimentClassifier(embed_dim=Config.embedding_dim, 
                       hidden_dim=Config.hidden_dim,
                       output_dim=Config.output_dim,
                       vocab_size=Config.vocab_size,
                       num_layer=Config.num_layer,
                       is_bidirectional=Config.is_bidirectional)

    random_tensor = torch.randint(low=0, 
                                  high=Config.vocab_size, 
                                  size=(64, Config.sequence_length), 
                                  dtype=torch.long)
    results = model(random_tensor)
    print(results.shape)

    # ----------------- traning model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=Config.lr)
    train_accuracies, test_accuracies, train_losses, test_losses = training(model=model, 
             criterion=criterion, 
             optimizer=optimizer, 
             train_dataloader=train_dataloader, 
             val_dataloader=val_dataloader,
             num_epochs=Config.num_epochs)
    # ------------------ save weight model
    save_model(model=model, path_model=Config.path_model)

    # ------------------ plot results
    plot_results(mode="Loss", 
              train_results=train_losses, 
              val_results=test_losses, 
              is_storage_results=True)
    plot_results(mode="Accuracy", 
              train_results=train_accuracies, 
              val_results=test_accuracies, 
              is_storage_results=True)
    
    # ------------------ evaluate 
    # model = load_model()
    metrics = evaluate(
        model=model,
        data_loader=test_dataloader,
        device='cpu',
        task_type='binary'
    )

    # In kết quả
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    res = inference(text="Tôi không thích quán ăn này, nó khá bẩn và phục vụ kém")
    print(res)


if __name__ == "__main__":
    res = inference(text="Tôi không thích quán ăn này, nó khá bẩn và phục vụ kém")
    print(res)