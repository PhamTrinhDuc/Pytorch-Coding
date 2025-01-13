import torch
import pandas as pd
import torch.nn as nn
from model import TransformersModel
from utils_data import VocabularyBuilder, PoemDataset

class Config:
    df = pd.read_csv("poem-datasets.csv")
    MAX_SEQ_LEN: int = 25
    TRAIN_BATCH_SIZE: int = 256
    vocab = VocabularyBuilder(df=df, max_seq_len=MAX_SEQ_LEN)
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIMS = 128
    HIDDEN_DIM = 128
    N_LAYERS = 2
    N_HEADS = 4
    DROPOUT = 0.2
    LR = 5.0
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformersModel(
    Config.VOCAB_SIZE, 
    Config.EMBEDDING_DIMS, 
    Config.N_HEADS, 
    Config.HIDDEN_DIM, 
    Config.N_LAYERS, 
    Config.DROPOUT
).to(Config.DEVICE)


def training(model, ):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95) # cập nhật lr = lr * 0.95 sau 1 epoch

    model.train()

    for epoch in range(Config.EPOCHS):
        losses = []
        