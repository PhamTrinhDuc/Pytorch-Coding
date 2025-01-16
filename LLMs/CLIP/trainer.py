import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import asdict
from clip import CLIP, CLIPLoss
from fashion_dataset import create_dataloader
from config import *

def train_epoch(model: nn.Module, 
                optimizer: torch.optim.AdamW, 
                criterion: callable, 
                train_loader: DataLoader, 
                device: str):
    train_loss = 0.0
    model.train()
    for idx, (sample) in enumerate(train_loader):
        image, caption, mask = (
            sample['image'].to(device), 
            sample['caption'].to(device), 
            sample['mask'].to(device)
        )
        # match with the order diagonal of the matrix contrastive [B, B]
        labels = torch.arange(len(image), dtype=torch.long).to(device)
        optimizer.zero_grad()

        contrastive_matrix = model(image, caption, mask) # tính outputs của model
        loss = criterion(logits=contrastive_matrix, labels=labels)
        train_loss += loss.item()

        #backward
        loss.backward()
        optimizer.step()

    epoch_loss = train_loss / len(train_loader)
    return epoch_loss


def evaluate_epoch(model: nn.Module, 
                   criterion: callable, 
                   val_loader: DataLoader, 
                   device: str):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            image, caption, mask = (
                sample['image'].to(device), 
                sample['caption'].to(device), 
                sample['mask'].to(device)
            )
            labels = torch.arange(len(image), dtype=torch.long).to(device)
            contrastive_matrix = model(image, caption, mask) # tính outputs của model

            loss = criterion(contrastive_matrix, labels)

            val_loss += loss.item()

    epoch_loss = val_loss / len(val_loader)
    return epoch_loss


def fit(model: nn.Module, 
        optimizer: callable, 
        criterion: torch.optim.AdamW, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int,
        path_model: str, 
        device: str):

    train_losses = []
    eval_losses = []
    best_loss_eval = 100
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        # trainning
        train_loss = train_epoch(model=model, 
                                            optimizer=optimizer, 
                                            criterion=criterion, 
                                            train_loader=train_loader, 
                                            device=device)
        train_losses.append(train_loss)

        # evaluate
        eval_loss = evaluate_epoch(model=model, 
                                             criterion=criterion, 
                                             val_loader=val_loader, 
                                             device=device)
        eval_losses.append(eval_loss)

        # save model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), path_model)
            best_loss_eval = eval_loss

        times.append(time.time() - epoch_start_time)
        print("-" * 60)
        print(
            "| end of epoch {:3d} | Time {:5.2f}s | Train Loss {:8.3f} "
            "| Val Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_loss, eval_loss
            )
        )
        print("-" * 60)

    model.load_state_dict(torch.load(f=path_model, 
                                     map_location=torch.device(device=device)))
    model.eval()
    metrics = {
        'train_losses': train_losses,
        'valid_losses': eval_losses,
        'time': times
    }
    return model, metrics


def main():
    args_data = ConfigDataset()
    train_loader, val_loader, test_loader = create_dataloader(**asdict(args_data))
    
    args_train = TrainingConfig()
    args_image = ViTConfig()
    args_text = TextEncoderConfig()
    model = CLIP(arg_image_encoder=args_image,
                 arg_text_encoder=args_text)
    
    optimizer = torch.optim.AdamW(params=model.parameters(), 
                                  lr=args_train.lr)
    
    model, metrics = fit(model=model, optimizer=optimizer, 
        criterion=CLIPLoss, 
        train_loader=train_loader, val_loader=val_loader, 
        num_epochs=args_train.epochs, path_model=args_train.path_model, 
        device=args_train.device)
    print(metrics)
    
    
if __name__ == "__main__":
    main()