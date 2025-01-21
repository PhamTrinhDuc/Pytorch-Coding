import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


class CatDogConfig:
    N_CLASSES: int=2
    IMG_SIZE: int=64
    ID2LABEL: dict={0: 'cat', 1: 'dog'}
    LABEL2ID: dict={"cat": 0, "dog": 1}
    NORMAL_MEAN = [0.485, 0.456, 0.406]
    NORMAL_STD = [0.229, 0.224, 0.225]

class ModelConfig:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_NAME = "retnest18"
    MODEL_WEIGHT = ROOT_DIR/'model'/'weights'/'best.pt'
    DEVICE = 'cpu'
