import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision import transforms
from datasets import load_dataset

class Config:
    DATASET_NAME: str='cats_vs_dogs'
    img_size: int=64
    n_classes: int=2
    test_size: float=0.2
    img_size: int=64 
    train_bs: int=512
    val_bs: int=256
    lr: float=1e-3
    weight_decay: float=1e-5
    num_epochs: int=10
    device: str="cuda" if torch.cuda.is_available() else "cpu"


class CatDogDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = self.data[index]['image']
        labels = self.data[index]['labels']

        if self.transform:
            images = self.transform(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels
    

def prepare_datasets(test_size: int, img_size: int, 
                      train_bs: int, val_bs: int):
    datasets = load_dataset(Config.DATASET_NAME)
    datasets = datasets['train'].train_test_split(test_size=test_size)
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485 , 0.456 , 0.406] ,
            [0.229 , 0.224 , 0.225]
        )
    ])
    train_dataset = CatDogDataset(data=datasets['train'], transform=image_transform)
    val_dataset = CatDogDataset(data=datasets['test'], transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False)

    return train_loader, test_loader


class CatDogModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        retnet_model = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(retnet_model.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = retnet_model.fc.in_features
        self.fc = nn.Linear(in_features=in_features, 
                            out_features=n_classes)
    
    def forward(self, x: torch.Tensor):
        # [B, C, H, W] => [B, 512, 1, 1]
        x = self.backbone(x)
        # [B, 512, 1, 1] => [B, 512]
        x = torch.flatten(x, 1)
        # [B, 512] => [1, 2]
        x = self.fc(x)
        return x
    

def training(model: CatDogModel, 
             optimizer, 
             criterion, 
             train_loader, 
             test_loader, 
             num_epochs: int,
             model_path: str="/best.pt",
             device: str='cuda'):

    for epoch in range(num_epochs):
        train_losses = []
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = sum(train_losses) / len(train_losses)
        
        val_losses = []
        best_val_loss = -1e9
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                if best_val_loss < loss:
                    best_val_loss = loss
                    torch.save(obj=model.state_dict(), f=model_path)
                
                val_losses.append(loss.item())
                
        val_loss = sum(val_losses) / len(val_losses)
        
        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.3f}\tVal loss: {val_loss:.3f}')


def main():
    config = Config()
    train_loader, test_loader = prepare_datasets(
        test_size=config.test_size, 
        img_size=config.img_size, 
        train_bs=config.train_bs,
        val_bs=config.val_bs,)

    model = CatDogModel(n_classes=config.n_classes).to(config.device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    # x = torch.randn(size=(1, 3, 224, 224))
    # output = model(x)
    # print(output.shape)

    training(model=model, 
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.num_epochs,
            device=config.device)

if __name__ == "__main__":
    main()