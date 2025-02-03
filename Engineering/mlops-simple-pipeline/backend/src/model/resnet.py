import torch.nn as nn
from typing import Literal
from torchvision import models

def create_retnest(n_classes: int=2, 
                   model_name: Literal["resnet18", "resnet34"] = "resnet18", 
                   load_pretrained: bool=False):
    
    if model_name == "resnet18":
        if load_pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18()
    elif model_name == "resnet34":
        if load_pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34()
    else:
        raise ValueError("Invalid model name. [resnet18, resnet34]")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features,
                         out_features=n_classes)
    return model
