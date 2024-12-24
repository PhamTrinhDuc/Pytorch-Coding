import pandas as pd
import numpy as np
import torch
import torchinfo
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class RNNModel(nn.Module):
    def __init__(self, 
                 embed_dim: int = 1, 
                 hidden_dim: int = 32,
                 output_dim: int = 1):
        super().__init__()
        """
        Args:
            embed_dim: 
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.model = nn.RNN(input_size=embed_dim, 
                            hidden_size=hidden_dim,
                            num_layers=2, # use 2 layer RNN
                            bidirectional=False, # don't use bidirectional
                            batch_first=True) # allow batch size in first position
        
        self.fully_connected = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim)
    
    def forward(self, x):
        rnn_output, hidden_output = self.model(x)
        last_hidden = hidden_output[:-1, :, :]
        # last_hidden = rnn_output[:, -1, :] # you can use it instead of the line above
        output = self.fully_connected(last_hidden)
        return output
    
    def show_model(self, 
                   batch_size: int, 
                   sequence_length: int):
        model = RNNModel(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim)
        print("Information of model: ")
        torchinfo.summary(model=model, input_size=(batch_size, sequence_length, self.embed_dim))


def create_sequence_data(data: np.ndarray, lag: int, ahead: int) -> Tuple[np.ndarray, np.ndarray]:
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

    X = []
    y = []
    for i in range(0, len(data) - lag - ahead + 1):
        X.append(data[i: i + lag]) # get lag sample for X
        y.append(data[i + lag: i + lag + ahead]) # get ahead sample for y 
    return np.array(X), np.array(y)


def preapare_data(data: np.ndarray, lag: int, ahead: int, train_ratio: float, batch_size: int):
    """
    Prepare the data for the Conv1D model training.

    Parameters:
    - data: The raw time series data.
    - lag: The number of time steps to use for predictions.
    - ahead: The number of time steps ahead to predict.
    - train_ratio: The ratio of the dataset to include in the train split.
    - batch_size: The size of the batch for the DataLoader.

    Returns:
    - train_test_loader: DataLoader for the training set.
    - X_test_tensor: PyTorch tensor for the test features.
    - y_test_tensor: PyTorch tensor for the test labels.
    """
    
    # get data transfomed
    X, y = create_sequence_data(data=data, lag=lag, ahead=ahead)
    X = X.reshape(X.shape[0], -1, 1)
    
    # split train test data
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]    
    # print(X_train.shape) => (num_sample_train, sequence_length, embed_dim)
    # print(y_train.shape) => (num_sample_test, output_dim)


    # convert array data to tensor data
    X_train_tensor = torch.tensor(data=X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(data=X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(data=y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(data=y_test, dtype=torch.float32)
    
    # init dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_test_loader, X_test_tensor, y_test_tensor


def training( 
            model: RNNModel, 
            criterion: nn.MSELoss, 
            optimizer: torch.optim.Adam, 
            train_test_loader: DataLoader, 
            num_epochs: int) -> Tuple[RNNModel, list]:
    """
    Train the RNN model.

    Parameters:
    - model: The PyTorch model to train.
    - criterion: The loss function.
    - optimizer: The optimization algorithm.
    - train_test_loader: DataLoader for the training set.
    - num_epochs: The number of epochs to train for.

    Returns:
    - model: The trained model.
    - losses: A list of loss values per epoch.
    """

    losses = []
    for num_epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for i, (sequences, labels) in enumerate(train_test_loader):
            optimizer.zero_grad()
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            # print(next(iter(sequences)).shape)  # => (batch, sequence_length, embed_dim)
            # print(next(iter(labels)).shape)

            # Forward pass
            y_pred = model(sequences)
            loss = criterion(y_pred, labels.unsqueeze(0)) # => thêm 1 chiều vào labels để tránh cảnh báo. Shape ban đầu: [64, 1], sau khi thêm: [1, 64, 1]


            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(train_test_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{num_epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
    return model, losses


def evaluate(model: RNNModel, 
             X_test_tensor: torch.tensor, 
             y_test_tensor: torch.tensor,
             output_dim: int):
    """
    Evaluate the MLP model.

    Parameters:
    - model: The PyTorch model to evaluate.
    - X_test_tensor: PyTorch tensor for the test features.
    - y_test_tensor: PyTorch tensor for the test labels.
    - ahead: The number of time steps ahead that the model predicts.

    Returns:
    - r2: R-squared score.
    - mae: Mean Absolute Error.
    - mse: Mean Squared Error.
    """

    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(DEVICE)

        predictions = model(X_test_tensor)
        predictions = predictions.view(-1, output_dim).cpu().detach().numpy()  # Reshape predictions to match y_test
        y_test = y_test_tensor.cpu().numpy()

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

    
    print(f'R2 Score: {r2}')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')

    return r2, mae, mse



def plot_results(model: RNNModel, 
                 X_test_tensor: torch.tensor,
                 y_test_tensor: torch.tensor,
                 file_path: str):
    
    X_test_tensor = X_test_tensor.to(DEVICE) 

    predictions = model(X_test_tensor).cpu().detach().numpy()
    y_test = y_test_tensor.numpy()

    plt.plot(predictions[: 500, 0], "r", label = "predictions")
    plt.plot(y_test[: 500, 0], "g", label = "y_test")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(file_path)
    plt.show()

def save_checkpoint(model: RNNModel, file_path: str):
    torch.save(obj=model, f=file_path)


class Config: 
    lag: int = 32
    ahead: int  = 1
    train_ratio: int = 0.8
    embed_dim: int = 1
    hidden_dim: int = 32
    output_dim: int = 1
    batch_size: int = 64
    num_epochs: int = 20


def main():

    # ------------------ show architecture model
    # model = RNNModel()
    # model.show_model(batch_size=2, sequence_length=64)

    # ------------------ Prepare dataset
    data = pd.read_csv("./temp.csv")["Temperature (C)"]
    train_test_loader, X_test_tensor, y_test_tensor = preapare_data(data=data,
                                                               lag=Config.lag, 
                                                               ahead=Config.ahead, 
                                                               train_ratio=Config.train_ratio, 
                                                               batch_size=Config.batch_size) 
    
    # ----------------- Training
    model = RNNModel(embed_dim=Config.embed_dim, hidden_dim=Config.hidden_dim, output_dim=Config.output_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    model, losses = training(model=model, 
                             criterion=criterion, 
                             optimizer=optimizer, 
                             train_test_loader=train_test_loader,
                             num_epochs=Config.num_epochs)
    
    save_checkpoint(model=model.to(DEVICE), file_path="./checkpoint_time_series.pth")
    
    plot_results(model=model, 
                 X_test_tensor=X_test_tensor, 
                 y_test_tensor=y_test_tensor, 
                 file_path="./rnn_time_series.png")
    
    r2, mae, mse = evaluate(model=model, 
                            X_test_tensor=X_test_tensor, 
                            y_test_tensor=y_test_tensor, 
                            output_dim=Config.output_dim)

if __name__ == "__main__":
    main()