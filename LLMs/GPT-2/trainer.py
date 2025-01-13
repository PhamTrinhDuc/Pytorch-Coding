import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils_data import (
    create_dataloader, 
    text_to_tokens_ids, 
    token_ids_to_text
)
from dataclasses import dataclass
from model import GPTModel, generate_text_simple


@dataclass
class GPTConfig124M:
    vocab_size: int = 50257
    context_length: int = 256
    d_model:int =  768
    ff_dim: int = d_model * 4
    max_new_tokens: int = 10
    num_heads: int = 12
    num_layers: int = 12
    drop_rate:float =  0.05
    qkv_bias: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ================
    lr: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 2
    weight_decay: float = 0.1


def generate_and_print_sample(model: GPTModel, 
                              tokenizer, 
                              start_context: str, 
                              context_length: int, 
                              max_new_tokens: int):
    model.eval()
    encoded = text_to_tokens_ids(text=start_context, tokenizer=tokenizer)
    with torch.no_grad():
        new_tokens_ids = generate_text_simple(
            model=model, 
            input=encoded, 
            max_new_tokens=max_new_tokens, 
            context_length=context_length
        )

        decoded = token_ids_to_text(tokens_ids=new_tokens_ids,
                                    tokenizer=tokenizer)
        print("Text generate: ", decoded.replace("\n", " "))


def trainer(model: GPTModel, 
            criterion: nn.CrossEntropyLoss, 
            optimizer: optim.AdamW, 
            train_loader: DataLoader, 
            epoch: int, 
            log_interval: int,
            device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    total = 0.0

    for idx, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device) # [N, sequence_len] 
        labels = labels.to(device) # [N, sequence_len]

        optimizer.zero_grad()
        # output: [N, vocab_size, sequence_len] => [N, vocab_size, seq_len]
        output = model(sequences).permute(0, 2, 1) 
        # print("shape output: ", output.shape) # [N, vocab_size, context_len]
        # print("Shape label: ", labels.shape) # [N, context_len]
        loss = criterion(output, labels)
        train_loss += loss.item()
        train_acc += (torch.argmax(input=output, dim=1) == labels).sum().item()
        total += len(labels)

        loss.backward()
        optimizer.step()

        if idx % log_interval == 0 and idx > 0:
            print(
                "| Epoch {:3d} | {}/{} batches | accuracy {:4.2f}".format(
                    epoch, idx, len(train_loader), train_acc / total
                )
            )
    epoch_acc = train_acc / total
    epoch_loss = train_loss / len(train_loader)
    return epoch_acc, epoch_loss


def evaluater(model: GPTModel, 
              criterion: nn.CrossEntropyLoss, 
              val_loader: DataLoader, 
              device):
    model.eval()
    val_loss, val_acc, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, (sequences, labels) in enumerate(val_loader):
            sequences = sequences.to(device) # [N, sequence_len] 
            labels = labels.to(device) # [N, sequence_len]

            # output: [N, vocab_size, sequence_len] => [N, vocab_size, seq_len]
            output = model(sequences).permute(0, 2, 1)
            loss = criterion(output, labels)

            val_loss += loss.item()
            val_acc += (torch.argmax(output, dim=1) == labels).sum().item()
            total += len(labels)

    epoch_loss = val_loss / total
    epoch_acc = val_acc / len(val_loader)
    return epoch_acc, epoch_loss


def fit(args: GPTConfig124M,
        start_context: str, 
        tokenizer,
        model: GPTModel, 
        criterion: nn.CrossEntropyLoss, 
        optimizer: optim.AdamW, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int, 
        device: str, 
        log_interval: int = 50, 
        path_model: str = "./checkpoint",):
    
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    times =  []
    best_loss_eval = 100


    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        train_acc, train_loss = trainer(model=model,
                                        criterion=criterion,
                                        optimizer=optimizer, 
                                        train_loader=train_loader,
                                        epoch=epoch, 
                                        log_interval=log_interval, 
                                        device=device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        eval_acc, eval_loss = evaluater(model=model, 
                                      criterion=criterion, 
                                      val_loader=val_loader, 
                                      device=device)
        valid_accuracies.append(eval_acc)
        valid_losses.append(eval_loss)

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
        generate_and_print_sample(model=model, 
                                  start_context=start_context,
                                  tokenizer=tokenizer,
                                  context_length=args.context_length, 
                                  max_new_tokens=args.max_new_tokens)
        print("-" * 60)

    model.load_state_dict(torch.load(f=path_model, 
                                     map_location=torch.device(device=device)))
    model.eval()
    metrics = {
        'train_accuracies': train_accuracies,
        'train_losses': train_losses,
        'valid_accuracies': valid_accuracies,
        'valid_losses': valid_losses,
        'time': times
    }
    return model, metrics


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
    torch.manual_seed(123)
    file_path = "LLMs/GPT-2/the-verdict.txt"

    with open(file_path, mode="r", encoding="utf-8") as f:
        data = f.read()
    
    ################ init model
    args = GPTConfig124M
    model = GPTModel(args=args)
    optimizer = optim.AdamW(params=model.parameters(), 
                            lr=args.lr, 
                            weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

    ################# setup data
    train_ratio = 0.9
    split_idx = int(train_ratio * len(data))

    train_loader = create_dataloader(
        text=data[:split_idx], 
        batch_size=args.batch_size,
        max_len=args.context_length, 
        stride=args.context_length, 
        is_shuffle=True, 
        is_drop_last=True, 
    )

    val_loader = create_dataloader(
        text=data[split_idx:], 
        batch_size=args.batch_size,
        max_len=args.context_length, 
        stride=args.context_length, 
        is_shuffle=False, 
        is_drop_last=False, 
    )

    ################## training
    start_context="Hello, I am"
    model, results = fit(args=args, 
                         start_context=start_context,
                         tokenizer=tokenizer, 
                         model=model, 
                         criterion=criterion, 
                         optimizer=optimizer, 
                         train_loader=train_loader, 
                         val_loader=val_loader, 
                         num_epochs=args.num_epochs, 
                         device=args.device)
    
    plot_result(num_epochs=args.num_epochs, 
                train_accs=results['train_accuracies'], 
                val_accs=results['valid_accuracis'], 
                train_losses=results['train_losses'], 
                val_losses=results['valid_losses'], 
                path_storage_result="./plot_results.png")

if __name__ == "__main__":
    main()
