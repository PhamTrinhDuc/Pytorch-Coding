import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_results(
        config,
        mode: str, 
        train_results: list, 
        val_results: list, 
        is_storage_results: bool = True): 
    if not isinstance(train_results, list) or not isinstance(val_results, list):
        raise ValueError("train_results and val_results must be list .")
    if len(train_results) != len(val_results):
        raise ValueError("train_results and val_results must be the same length .")
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(train_results, label=f"Train {mode}", color='blue')
    ax[0].set_title(f"Training {mode}")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("{mode}")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(val_results, label=f"Validation {mode}", color='orange')
    ax[1].set_title(f"Validation {mode}")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(f"{mode}")
    ax[1].grid(True)
    ax[1].legend()
    if is_storage_results:
        storage_results = config.path_result.format(mode=mode)
        try:
            plt.savefig(storage_results)
        except Exception as e:
            print(f"Error while store results: {e}")

    plt.show()


def save_model(model: nn.Module, path_model) -> None:
    torch.save(obj=model.state_dict(), 
               f=path_model)

def load_model(model: nn.Module, path_model: str, device) -> nn.Module:
    model.load_state_dict(torch.load(f=path_model, 
                                     map_location=torch.device(device=device)))
    return model