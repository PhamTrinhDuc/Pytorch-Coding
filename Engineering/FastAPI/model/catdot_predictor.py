import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from config import CatDogConfig, ModelConfig
from logs.logger import Logger

LOGGER = Logger(name=__file__, log_file="predictor.log")
LOGGER.log.info("Starting model serving")

class CatDogModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        retnet_model = resnet18()
        self.backbone = nn.Sequential(*list(retnet_model.children())[:-1])
        for param in self.backbone.parameters("IMAGENET1K_V1"):
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


class CatDogPredictor:
    def __init__(self, 
                 model_name: str, 
                 model_path: str, 
                 device: str='cpu'):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.catdog_cfg = CatDogConfig()

        self.model = self._load_model()
        self.transform = self._create_transform()

    def _load_model(self):
        try:
            model = CatDogModel(n_classes=2)
            model.load_state_dict(
                state_dict=torch.load(f=self.model_path, 
                                      map_location=torch.device('cpu')),
            )
            model.eval()
            return model
        except Exception as e:
            LOGGER.log.error("Load model failed")
            LOGGER.log.error(f"Error: {e}")
            return None

    def _create_transform(self):
        transform = transforms.Compose([
            transforms.Resize(size=(self.catdog_cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.catdog_cfg.NORMAL_MEAN,
                std=self.catdog_cfg.NORMAL_STD
            )
        ])
        return transform
    
    def output2pred(self, output: torch.Tensor):
        probabilities = F.softmax(input=output, dim=1)
        
        best_prob = torch.max(probabilities, dim=1)[0].item()
        best_id = torch.max(probabilities, dim=1)[1].item()

        predicted_cls = self.catdog_cfg.ID2LABEL[best_id]
        return probabilities, best_prob, best_id, predicted_cls
    
    async def _model_inference(self, input: torch.Tensor):
        with torch.no_grad():
            output = self.model(input.to(self.device)).cpu()
        
        return output
    
    async def predict(self, image):
        image = Image.open(image)
        if image.mode == 'RGBA':
            image = image.convert(mode="RGB")
        
        transform_img = self.transform(image)
        transform_img = transform_img.unsqueeze(0) # add batch dimension
        output = await self._model_inference(transform_img)
        probs, best_probs, predicted_id, predicted_cls = self.output2pred(
            output=output
        )

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_probs, predicted_id, predicted_cls)

        torch.cuda.empty_cache()

        resp_dict = {
            "probs": probs,
            "best_probs": best_probs,
            "predicted_class" : predicted_cls,
            "predictor_name" : self.model_name
        }
        return resp_dict

def main():
    config = ModelConfig()
    predictor = CatDogPredictor(model_name=config.MODEL_NAME, 
                                model_path=config.MODEL_WEIGHT,
                                device=config.DEVICE)

    output = predictor.predict(image_path="./images/golden_dog.png")
    print(output)

if __name__ == "__main__":
    main()