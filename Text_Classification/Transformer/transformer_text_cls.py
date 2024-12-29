import numpy as np
import pandas as pd
import torch
import re
import time
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
    num_epochs: int = 1
    lr: float = 0.001

    path_model: str = "./Text_Classification/Transformer/checkpoint/transformer_tex_cls.pth"
    path_result: str = "./Text_Classification/Transformer/results/transformer_text_cls.png"


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
    df_train = pd.read_csv("data/cls_text/train.csv")
    df_val = pd.read_csv("data/cls_text/val.csv")
    df_test = pd.read_csv("data/cls_text/test.csv")

    def __init__(self):
        self.train_data, self.val_data, self.test_data = self._prepare_data()
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab()
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

    def _prepare_data(self) -> tuple[Dataset, Dataset, Dataset]:
        
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

        self.df_train['sentence'].apply(preprocess_text)
        self.df_val['sentence'].apply(preprocess_text)
        self.df_test['sentence'].apply(preprocess_text)
        
        return (
            Dataset.from_pandas(self.df_train),
            Dataset.from_pandas(self.df_val),  
            Dataset.from_pandas(self.df_test)
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
                       seq_length: int = Config.sequence_length) :
        
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


def train_epoch(model: nn.Module, 
                optimizer: torch.optim.AdamW, 
                criterion: nn.CrossEntropyLoss, 
                train_loader: DataLoader, 
                device: str, 
                epoch: int, 
                log_interval: int = 50):
    
    model.train()
    train_acc , train_loss = 0.0 , 0.0
    total = 0.0
    start_time = time.time()

    for idx, (sentence, label) in enumerate(train_loader):
        sentence = sentence.to(device)
        labels = label.to(device)
        optimizer.zero_grad()

        outputs = model(sentence) # tính outputs của model
        loss = criterion(outputs, labels) # tính loss
        train_loss += loss.item()

        #backward
        loss.backward()
        optimizer.step()

        train_acc += (torch.argmax(outputs, 1) == labels).sum().item()
        total += len(labels)

        if idx % log_interval == 0 and idx > 0:
            print(
                "| Epoch {:3d} | {}/{} batches | accuracy {:4.2f}".format(
                    epoch, idx, len(train_loader), train_acc / total
                )
            )
    epoch_acc = train_acc / total
    epoch_loss = train_loss / len(train_loader)
    return epoch_acc, epoch_loss


def evaluate_epoch(model: nn.Module, 
                   criterion: nn.CrossEntropyLoss, 
                   val_loader: DataLoader, 
                   device: str):
    val_acc, val_loss = 0.0, 0.0
    total = 0.0

    model.eval()
    with torch.no_grad():
        for idx, (sentences, labels) in enumerate(val_loader):
            sentences = sentences.to(device)
            labels = labels.to(device)

            outputs = model(sentences)
            loss = criterion(outputs, labels)

            val_acc += (torch.argmax(outputs, 1) == labels).sum().item()
            val_loss += loss.item()
            total += len(labels)

    epoch_acc = val_acc / total
    epoch_loss = val_loss / len(val_loader)
    return epoch_acc, epoch_loss


def fit(model: nn.Module, 
            optimizer: nn.CrossEntropyLoss, 
            criterion: torch.optim.AdamW, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            num_epochs: int,
            path_model: str, 
            device: str):

    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        # trainning
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_loader, device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        # evaluate
        eval_acc, eval_loss = evaluate_epoch(model, criterion, val_loader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # save model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), path_model)
            best_loss_eval = eval_loss

        times.append(time.time() - epoch_start_time)
        print("-" * 60)
        print(
            "| end of epoch {:3d} | Time {:5.2f}s | Train Acc {:8.3f} | Train Loss {:8.3f} "
            "| Val Acc {:8.3f} | Val Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 60)

    model.load_state_dict(torch.load(f=path_model, 
                                     map_location=torch.device(device=device)))
    model.eval()
    metrics = {
        'train_accuracies': train_accs,
        'train_losses': train_losses,
        'valid_accuracies': eval_accs,
        'valid_losses': eval_losses,
        'time': times
    }
    return model, metrics


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


def inference(
        text: str,
        model: nn.Module, 
        path_model: str, 
        device):
    model.load_state_dict(torch.load(f=path_model, 
                                map_location=torch.device(device=device)))
    
    processor = ProcessingData()
    vocab = processor.vocab
    text_pipelie = processor.text_pipeline

    text_list = []
    text_processed = text_pipelie(text)[:Config.sequence_length]
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


def plot_result(num_epochs: int, 
                train_accs: list, 
                val_accs: list, 
                train_losses: list, 
                val_losses:list,
                path_storage_result: str = None):
    
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_accs, label = 'Training')
    axs[0].plot(epochs, val_accs, label = 'EValuation')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Acccuracy")

    axs[1].plot(epochs, train_losses, label = 'Training')
    axs[1].plot(epochs, val_losses, label = 'Evaluation')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    if path_storage_result:
        plt.savefig(path_storage_result)
    
    plt.legend()


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

    # mock_data = torch.randint(low=0, high=2, size=(Config.batch_size, Config.sequence_length))
    # output = classifier(mock_data)
    # print(output.size())


    # ---------------------- Get data
    processor = ProcessingData()
    train_loader, val_loader, test_loader = processor.create_dataloader()

    # ---------------------- training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=Config.lr)
    model, metrics = fit(model=classifier, optimizer=optimizer, criterion=criterion, 
        train_loader=train_loader, val_loader=val_loader, num_epochs=Config.num_epochs, 
        path_model=Config.path_model, device=DEVICE)
    
    # ---------------------- save results
    plot_result(num_epochs=Config.num_epochs, 
                train_accs=metrics['train_accuracies'], 
                val_accs=metrics['valid_accuracies'], 
                train_losses=metrics['train_losses'],
                val_losses=metrics['val_losses'], 
                path_storage_result=Config.path_result)
    
    # ---------------------- evaluate with metrics 
    metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device='cpu',
        task_type='binary'
    )

    # In kết quả
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # ---------------------- inference
    results = inference(text="Quán ăn khá sạch sẽ, tôi sẽ quay lại vào lần sau",
                        model=classifier, path_model=Config.path_model, device=DEVICE)
    print(results)
if __name__ == "__main__":
    main()