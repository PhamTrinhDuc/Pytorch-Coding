import numpy as np
import pandas as pd
import torchinfo
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torchtext.data.utils import get_tokenizer


class LSTMTimeSeries(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 hiddent_dim: int,
                 output_dim: int,
                 num_layer: int = 1,
                 is_bidirectional: bool = False):
        super().__init__()
        self.model = nn.LSTM(input_size=embed_dim, 
                             hidden_size=hiddent_dim,
                             num_layers=num_layer,
                             bidirectional=is_bidirectional)
        
        self.fc = nn.Linear(in_features=hiddent_dim,
                            out_features=output_dim)
        

    def forward(self, x):
        pass 