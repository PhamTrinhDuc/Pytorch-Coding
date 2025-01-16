import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from text_encoder import TextEncoder, TextEncoderRetrieval 
from image_encoder import ViTEncoder
from config import TextEncoderConfig, ViTConfig


class CLIP(nn.Module):
    def __init__(self, 
                 arg_text_encoder: TextEncoderConfig, 
                 arg_image_encoder: ViTConfig):
        super().__init__()
    
        self.vision_encoder = ViTEncoder(**asdict(arg_image_encoder))

        self.text_encoder = TextEncoder(**asdict(arg_text_encoder))

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image: torch.Tensor, text: torch.Tensor, mask=None):
        # image: [N, C, H, W], text: [N, seq_len]

        I = self.vision_encoder(image) # [B, embedding_image]
        T = self.text_encoder(text, mask) # [B, embedding_text = embedding_image]

        # [B, embed_dim] @ [embed_dim, B] = [B, B] (See contrastive images in folder images)
        contrastive_matrix = torch.matmul(I, T.transpose(-2, -1)) * torch.exp(self.temperature)
        
        return contrastive_matrix


def CLIPLoss(logits: torch.Tensor, labels: torch.Tensor):
        """
        Contrastive loss, See contrastive images in folder images
        logits: The matrix corresponds to the viewing angle from image to text and text to image

        >>>> Examples
        logits = [
            [0.9, 0.2, 0.1],   point corresponds of image1 with text1, text2, text3
            [0.1, 0.8, 0.3],   point corresponds of image2 with text1, text2, text3 
            [0.2, 0.1, 0.7]    point corresponds of image3 with text1, text2, text3
        ]
        labels = [0, 1, 2] # indicates that image1 matches text1, image2 matches text2, image3 matches text3
        
        >>>> Apply Softmax function
        probabilities = [
            [0.6, 0.2, 0.2],   probability of image1 matching each text
            [0.2, 0.6, 0.2],   probability of image2 matching each text
            [0.2, 0.2, 0.6]    probability of image3 matching each text
        ]

        >>>> Caculate cross entropy:
        one-hot labels: 
        labels = [
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1]
        ]

        loss = -sum(i=1->n)sum(j=1->n)(labels[i][j] * log(probabilities[i][j]))
             = - log(0.6) - log(0.6) - log(0.6)
        """
        # labels match with the order diagonal of the matrix
        
        # Change the view from image->text to text->image
        # For each text, the corresponding image must have the highest score
        loss_T_to_I = F.cross_entropy(input=logits.transpose(-2, -1), target=labels)
        # The view from image->text
        # For each image, the corresponding text must have the highest score
        loss_I_to_T = F.cross_entropy(input=logits, target=labels)

        loss = (loss_T_to_I + loss_I_to_T) / 2 
        return loss


def main():
    args_image = ViTConfig()
    args_text = TextEncoderConfig()
    clip = CLIP(arg_image_encoder=args_image, 
                arg_text_encoder=args_text, 
                is_retrieval=False)
    mock_image = torch.randn(size=(5, 3, 128, 128))
    mock_text = torch.randint(0, 10, size=(5, 32))
    contrastive_matrix = clip(mock_image, mock_text)

    labels = torch.arange(mock_image.shape[0]).to(device="cpu")
    loss = CLIPLoss(contrastive_matrix, labels)
    print(loss)


if __name__ == "__main__":
    main()