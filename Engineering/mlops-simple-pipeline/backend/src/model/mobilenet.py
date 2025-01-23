import torch.nn as nn
from typing import Literal
from torchvision import models

def create_retnest18(n_classes: int=2, 
                     model_name: Literal["mobilenet_v2", "mobilenet_v3_small"] = "mobilenet_v2", 
                     load_pretrained: bool=False):
    if model_name == "mobilenet_v2":
        if load_pretrained:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v2()
    elif model_name == "mobilenet_v3_small":
        if load_pretrained:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v3_small()
    else:
        raise ValueError("Invalid model name. [mobilenet_v2, mobilenet_v3_small]")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features,
                                    out_features=n_classes)
    return model
