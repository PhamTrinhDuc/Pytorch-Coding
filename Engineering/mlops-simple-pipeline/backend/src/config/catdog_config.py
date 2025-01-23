from dataclasses import dataclass


@dataclass
class CatDogArgs:
    n_classes: int = 2
    img_size: int = 224
    classes = ["cat", "dog"]
    id2label = {0: "cat", 1: "dog"}
    label2id = {"cat": 0, "dog": 1}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]