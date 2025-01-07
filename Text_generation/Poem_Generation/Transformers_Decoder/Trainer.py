import torch
import pandas as pd
import torch.nn as nn
from Vectorization import Vocab
from model import TransformersModel


class Config:
    df = pd.read_csv("poem-datasets.csv")
    vocab = Vocab(df)
    vocab = vocab.build_vocab()

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


input_tests = torch.randint(1, 10, (1, 10)).to(Config.DEVICE)
with torch.no_grad():
    output = model(input_tests)
    print(output.shape)


def training():
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95) # cập nhật lr = lr * 0.95 sau 1 epoch

    model.train()

    for epoch in range(Config.EPOCHS):
        losses = []
        