import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.optim import AdamW
from encoder import VaeEncoder
from decoder import VaeDecoder

class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int=4):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = VaeEncoder(in_encode=in_channels)
        self.decoder = VaeDecoder(out_decode=in_channels)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        x_latent, mu, log_variance = self.encoder(x)
        x_reconstructed = self.decoder(x_latent)

        return x_reconstructed, mu, log_variance
    

def VaeLoss(x: torch.Tensor, 
            x_reconstructed: torch.Tensor, 
            mu: torch.Tensor, 
            log_variance: torch.Tensor):
    
    # 1. Reconstruction Loss
    recon_loss = nn.BCELoss(reduction="sum")(F.sigmoid(x_reconstructed), x)
    # 2. KL Divergence Loss
    # KL(N(μ, σ) || N(0, 1)) = 0.5 * Σ(μ² + σ² - ln(σ²) - 1)
    kl_loss = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
    total_loss = recon_loss + kl_loss

    return total_loss


def create_dataloader(batch_size: int):
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    trainset = MNIST(root='./data/', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root='./data/', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def fit(model: VAE, 
        optimizer:AdamW, 
        criterion, 
        train_loader: DataLoader, 
        test_loader: DataLoader,
        num_epochs: int, device: str="cpu"):

    model.train()    
    for batch_idx, (epoch, _) in enumerate(range(num_epochs)):
        loss_epoch = 0.0
        for image, _ in train_loader:
            optimizer.zero_grad()
            image = image.to(device)
            x_reconstructed, mu, log_variance = model(image)
            loss = criterion(x=image, x_reconstructed=x_reconstructed,
                           mu=mu, log_variance=log_variance)
            
            loss_epoch += loss.item()

            loss.backward()
            optimizer.step()
        
        print("\tEpoch", epoch + 1, "complete!", "\t Loss Epoch: ", loss_epoch)


def main():
    batch_size = 1
    in_channels = 3
    W_image = H_image = 224
    batch_size = 32
    lr = 0.05
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = create_dataloader(batch_size=batch_size)

    model = VAE(in_channels=in_channels)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    fit(model=model, optimizer=optimizer, 
        criterion=VaeLoss, train_loader=train_loader, test_loader=val_loader, 
        num_epochs=10, device=device)


if __name__ == "__main__":
    main()