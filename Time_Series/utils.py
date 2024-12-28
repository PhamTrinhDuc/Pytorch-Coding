import torch
import torch.nn as nn
import numpy as np
import matplotlib as plt

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
    
    return np.array(features), np.array(labels)

def save_model(model: nn.Module, file_path: str):
    torch.save(obj=model, f=file_path)

def load_model(model: nn.Module,
               file_path: str, 
               device: str = "cpu") -> nn.Module:
    model.load_state_dict(torch.load(f=file_path, 
                          map_location=torch.device(device)))
    return model


def plot_results(model: nn.Module, 
                 X_test_tensor: torch.tensor,
                 y_test_tensor: torch.tensor,
                 file_path: str, 
                 device):
    
    X_test_tensor = X_test_tensor.to(device) 

    predictions = model(X_test_tensor).cpu().detach().numpy()
    predictions = predictions.squeeze(0)
    y_test = y_test_tensor.numpy()

    plt.plot(predictions[: 500, 0], "r", label = "predictions")
    plt.plot(y_test[: 500, 0], "g", label = "y_test")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(file_path)
    plt.show()


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

    plt.plot(times,y_to_plot, color="steelblue",marker="o", label="True value")
    plt.plot(times, pred_to_plot, color="orangered", marker="X", label="Prediction")

    plt.title("Temperature in every hours")
    plt.xlabel("Hour")
    plt.ylabel("Temperate (C)")
    plt.legend()
    if storage_path:
        plt.savefig(storage_path)
    plt.show()