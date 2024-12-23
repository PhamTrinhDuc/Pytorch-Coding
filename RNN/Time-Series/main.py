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

DEVICE = "0" if torch.cuda.is_available() else "cpu"


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
        
        self.fully_connected = nn.Linear(in_features=hidden_dim,
                                out_features=output_dim)
    
    def forward(self, x):
        rrn_output, hidden_output = self.model(x)
        last_hidden = hidden_output[:-1, :, :]
        # last_hidden = rrn_output[:, -1, :] # you can use it instead of the line above
        output = self.fully_connected(last_hidden)
        return output
    
    def show_model(self, 
                   batch_size: int, 
                   sequence_length: int):
        model = RNNModel(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim)
        print("Information of model: ")
        torchinfo.summary(model=model, input_size=(batch_size, sequence_length, self.embed_dim))
    


class Pipeline:
    def __init__(self):
        pass

    def _create_sequence_data(self, data: np.ndarray, lag: int, ahead: int) -> Tuple[np.ndarray, np.ndarray]:
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
        for i in range(0, len(data)):
            X.append(data[i: i + lag]) # get lag sample for X
            y.append(data[i + lag: i + lag + ahead]) # get ahead sample for y 
        return np.array(X), np.array(y)

    def preapare_data(self, data: np.ndarray, lag: int, ahead: int, test_ratio: float, batch_size: int):
        """
        Prepare the data for the Conv1D model training.

        Parameters:
        - data: The raw time series data.
        - lag: The number of time steps to use for predictions.
        - ahead: The number of time steps ahead to predict.
        - train_ratio: The ratio of the dataset to include in the train split.
        - batch_size: The size of the batch for the DataLoader.

        Returns:
        - train_loader: DataLoader for the training set.
        - X_test_tensor: PyTorch tensor for the test features.
        - y_test_tensor: PyTorch tensor for the test labels.
        """
        
        # get data transfomed
        X, y = self._create_sequence_data(data=data, lag=lag, ahead=ahead)
        
        # split train test data
        X_train, y_train, X_test, y_test = train_test_split((X, y), test_size=test_ratio, random_state=42)
        
        # convert array data to tensor data
        X_train_tensor = torch.tensor(data=X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(data=X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(data=y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(data=y_test, dtype=torch.float32)
        
        # init dataloader
        X_train_loader = DataLoader(dataset=X_train_tensor, batch_size=batch_size, shuffle=True, num_workers=2)
        y_train_loader = DataLoader(dataset=y_train_tensor, batch_size=batch_size, num_workers=2)
        
        # you can use it instead of 2 lines above
        # train_data = TensorDataset(X_train_tensor, y_train_tensor)
        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        return X_train_loader, y_train_loader, X_test_tensor, y_test_tensor


    def training(self, 
                 model: RNNModel, 
                 criterion: nn.MSELoss, 
                 optimizer: torch.optim.Adam, 
                 train_loader: DataLoader, 
                 test_loader: DataLoader, 
                 num_epochs: int):
        """
        Train the RNN model.

        Parameters:
        - model: The PyTorch model to train.
        - criterion: The loss function.
        - optimizer: The optimization algorithm.
        - train_loader: DataLoader for the training set.
        - num_epochs: The number of epochs to train for.

        Returns:
        - model: The trained model.
        - losses: A list of loss values per epoch.
        """

        losses = []
        for num_epoch in range(num_epochs):
            model.train()
            running_loss = 0
            for i, (sequences, label) in enumerate(train_loader, test_loader):
                optimizer.zero_grad()

                # Forward pass
                y_pred = model(sequences)
                loss = criterion(y_pred, label)


                # Backward and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            losses.append(epoch_loss)
            print(f'Epoch [{num_epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            
        return model, losses



    def evaluate():
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

        pass

    def plot_results():
        pass 


if __name__ == "__main__":
    model = RNNModel()
    model.show_model(batch_size=2, sequence_length=64)