import numpy as np
import pandas as pd
import torchinfo
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error
)
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class Config: 
    embed_dim: int = 1
    hidden_dim: int = 32
    output_dim: int = 1
    num_layer: int = 2
    is_bidirectional: bool = True
    lag: int = 32 # 32 front temperatures
    ahead: int = 1 # predict 1 temperature later
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    dropout_prob: float = 0.2
    path_model: str = "./LSTM/checkpoint/time_series.pth"
    path_results_loss: str = "./LSTM/results/loss.png"


class LSTMTimeSeries(nn.Module):
    def __init__(self, 
                 embed_dim: int = 1, 
                 hidden_dim: int = 32,
                 output_dim: int = 1,
                 dropout_prob: float = 0.2,
                 num_layer: int = 1,
                 is_bidirectional: bool = False):
        super().__init__()
        self.model = nn.LSTM(input_size=embed_dim, 
                             hidden_size=hidden_dim,
                             num_layers=num_layer,
                             bidirectional=is_bidirectional)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=output_dim)

    def forward(self, x):
        output_lstm, (hidden_lstm, cell_lstm) = self.model(x)
        last_hidden = hidden_lstm[-1, :, ] # hidden_lstm: [num_layer, batch_size, hidden_dim]
        # last_hidden = output_lstm[:, -1, :] # output_lstm: [batch_size, seq_len, hidden_dim]
        last_hidden_norm = self.norm(self.dropout(last_hidden))
        output = self.fc(last_hidden_norm)
        return output


def create_sequence_data(
        df: np.ndarray, 
        lag: int,
        ahead: int):
    """
    Create sequences from time series data for use in time series forecasting.

    Parameters:
    - data (array-like): The time series data as a list or numpy array.
    - lag (int): The number of data points in each input sequence.
    - ahead (int): The number of data points to predict in the future.

    Returns:
    - X (numpy.ndarray): A 2D array of shape (num_sequences, lag) containing the input sequences.
    - y (numpy.ndarray): A 2D array of shape (num_sequences, ahead) containing the corresponding labels.
    """
    features = []
    labels = []

    for i in range(0, len(df) - lag - ahead):
        features.append(df[i: i + lag])
        labels.append(df[i+lag: i + lag + ahead])
    
    return features, labels


class TemperatureDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        if self.transform:
            X = self.transform(X)
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

    
def prepare_data(data: np.ndarray, 
                 lag: int,
                 ahead: int,
                 batch_size: int = 8,
                 test_ratio: float = 0.1,
                 val_ratio: float = 0.2):
    # create sequence data features and labels
    X, y = create_sequence_data(df=data, lag=lag, ahead=ahead)

    # create train test for data time series
    test_size = int(test_ratio * len(X))
    val_size = int(val_ratio * len(X))
    train_size = len(X) - test_size - val_size
    X_train, X_val, X_test = X[:train_size], X[train_size: train_size + val_size], X[train_size + val_size: ]
    y_train, y_val, y_test = y[:train_size], y[train_size: train_size + val_size], y[train_size + val_size: ]
    
    # init data with custom dataset
    train_dataset  = TemperatureDataset(X=X_train, y=y_train)
    val_dataset = TemperatureDataset(X=X_val, y=y_val)
    test_dataset = TemperatureDataset(X=X_test, y=y_test)
    
    # create data loader
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=2)
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=2)
    test_dataloader = DataLoader(dataset=test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 num_workers=2)
    
    return train_dataloader, val_dataloader, test_dataloader


def evaluate(model: LSTMTimeSeries, 
             data_loader: DataLoader,
             output_dim: int) -> float:
    
    X_tensor, y_tensor = (
        torch.tensor(np.array(data_loader.dataset.X), dtype=torch.float32), 
        torch.tensor(np.array(data_loader.dataset.y), dtype=torch.float32)
    )
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(DEVICE)
        predictions = model(X_tensor).view(-1, output_dim).cpu().detach().numpy() # Reshape predictions to match y_test
        labels = y_tensor.cpu().numpy()

    r2 = r2_score(y_true=labels, y_pred=predictions)
    mse = mean_squared_error(y_true=labels, y_pred=predictions)
    mae = mean_absolute_error(y_true=labels, y_pred=predictions)
    return r2, mse, mae


def fit(model: LSTMTimeSeries, 
        criterion: nn.MSELoss,
        optimizer: torch.optim.AdamW,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int) -> tuple[list, list]:
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # training model with trainig set
        batch_train_losses = []
        model.train()
        for i, (sequences, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(sequences)
            loss = criterion(predictions, labels)
            batch_train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
        
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_losses)


        # evaluate model with validation set 
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                batch_val_losses.append(loss.item())
        val_loss = sum(batch_val_losses) / len(batch_val_losses)
        val_losses.append(val_loss)

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')
    return train_losses, val_losses


def inference(model_path: str, input_data, device: str = "cpu"):
    """
    Load trained LSTM model and perform inference
    
    Args:
        model_path (str): Path to saved model weights
        input_data (numpy.ndarray): Input data for inference
        device (str): Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        numpy.ndarray: Model predictions
    """

    model = load_model(file_path=model_path).to(device=device)

    # convert input to tensor and reshape if needed
    if isinstance(input_data, np.ndarray):
        input_data = torch.FloatTensor(input_data).to(device=device)

    elif isinstance(input_data, torch.Tensor):
        input_data = input_data.to(device=device)

    # Add bach dimension if not present
    if len(input_data) == 2:
        input_data = input_data.unsqueeze(0)
        
    model.eval()
    with torch.no_grad():
        predictions = model(input_data)
    predictions = predictions.cpu().detach().numpy()
    return predictions


def plot_loss(train_losses: list, val_losses: list, storage_results:str = None): 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax[0].plot(train_losses)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(val_losses)
    ax[1].set_title("Evaluate Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")

    if storage_results:
        plt.savefig(storage_results)
    plt.show()


def plot_difference(y_pred: list, y: list, storage_path: str = None):
    plt.plot(figsize =(20, 6))
    times = range(len(y))
    y_to_plot = y.flatten()
    pred_to_plot = y_pred.flatten()

    plt.plot(times,y_to_plot, color="steeblue",marker="o", label="True value")
    plt.plot(times, pred_to_plot, color="orangered", marker="X", label="Prediction")

    plt.title("Temperature in every hours")
    plt.xlabel("Hour")
    plt.ylabel("Temperate (C)")
    plt.legend()
    if storage_path:
        plt.savefig(storage_path)
    plt.show()


def save_model(model: LSTMTimeSeries, path_model: str) -> None:
    torch.save(obj=model.state_dict(), 
               f=path_model)


def load_model(path_model: str) -> LSTMTimeSeries:
    model = LSTMTimeSeries(embed_dim=Config.embed_dim, 
                           hidden_dim=Config.hidden_dim,
                           output_dim=Config.output_dim,
                           dropout_prob=Config.dropout_prob,
                           num_layer=Config.num_layer,
                           is_bidirectional=Config.is_bidirectional)
    
    model.load_state_dict(torch.load(f=Config.path_model, 
                                     map_location=torch.device(DEVICE)))
    return model


def main():
    # ---------------------------- show architecture model
    model = LSTMTimeSeries(embed_dim=Config.embed_dim, 
                           hidden_dim=Config.hidden_dim,
                           output_dim=Config.output_dim,
                           dropout_prob=Config.dropout_prob,
                           num_layer=Config.num_layer,
                           is_bidirectional=Config.is_bidirectional)
    torchinfo.summary(model=model, input_size=(1, Config.lag, Config.embed_dim))

    # ---------------------------- prepare dataset
    data = pd.read_csv("data/temp.csv")["Temperature (C)"]
    train_dataloader, val_dataloader, test_dataloader = prepare_data(data=data,
                                                                     lag=Config.lag,
                                                                     ahead=Config.ahead,
                                                                     batch_size=Config.batch_size,
                                                                     test_ratio=Config.test_ratio,
                                                                     val_ratio=Config.val_ratio)
    # ----------------------------- training 
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=Config.lr)
    train_losses, val_losses = fit(model=model, criterion=criterion, optimizer=optimizer, 
                                   train_loader=train_dataloader,
                                   val_loader=val_dataloader, 
                                   num_epochs=Config.num_epochs)
    
    # ------------------------------ plot loss train and loss validation
    plot_loss(train_losses=train_losses, val_losses=val_losses, 
              storage_results=Config.path_results_loss)
    
    # ------------------------------ validation on r2, mse, mae metrics
    r2_val, mse_val, mae_val = evaluate(model=model, data_loader=val_dataloader, output_dim=Config.output_dim)
    r2_test, mse_test, mae_test = evaluate(model=model, data_loader=test_dataloader, output_dim=Config.output_dim)
    print(f"R2 val: {r2_val} \t MSE val: {mse_val} \t MAE val: {mae_val}")
    print(f"R2 test: {r2_test} \t MSE test: {mse_test} \t MAE test: {mae_test}")

    # ------------------------------ plot results after inference with test set
    X_test, y_test = test_dataloader.dataset.X, test_dataloader.y
    inputs = torch.tensor(X_test[:500], dtype=torch.float32).to(device=DEVICE)
    outputs = inference(model_path=Config.path_model, input_data=inputs)
    plot_difference(y=y_test[:500], y_pred=outputs[:500])

if __name__ == "__main__":
    main()