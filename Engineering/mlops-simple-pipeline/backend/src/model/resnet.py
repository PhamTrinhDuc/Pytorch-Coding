import torch.nn as nn
from typing import Literal
from torchvision import models

def create_retnest18(n_classes: int=2, 
                     model_name: Literal["retnest18", "retnest34"] = "retnest18", 
                     load_pretrained: bool=False):
    if model_name == "retnest18":
        if load_pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18()
    elif model_name == "retnest34":
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
