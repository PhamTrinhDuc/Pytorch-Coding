import os
import torch
import tiktoken
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from text_encoder import tokenization
    

class FashionDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame, 
                 tokenizer: callable, 
                 target_size: tuple[int, int]):
        super().__init__
        self.tokenizer = tokenizer
        self.dataframe = data_frame[data_frame['subCategory'].str.lower() != 'innerwear']
        self.target_size = target_size  # Desired size for the square image

        self.class_names = [str(name).lower() 
                            for name in self.dataframe['subCategory']]
        for i, class_name in enumerate(self.class_names):
            if class_name == "lips": 
                self.class_names[i] = "lipstick"
            elif class_name == "eyes":
                self.class_names[i] = "eyelash"
            elif class_name == "nails":
                self.class_names[i] = "nail polish"
        self.dataframe['subCategory'] = self.class_names

        self.captions = {idx: class_name for idx, class_name in enumerate(self.class_names)}

        self.transform =  T.Compose([
            T.ToTensor(), 
            T.Resize(size=target_size)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        while True:
            sample = self.dataframe.iloc[index]
            img_path = os.path.join("data/images", f"{sample['id']}.jpg")
            try:
                # Attempt to open the image
                image = Image.open(img_path).convert("RGB")

            except (FileNotFoundError, IOError):
                # If the image is not found, skip this sample by incrementing the index
                index = (index + 1) % len(
                    self.dataframe
                )  # Loop back to the start if we reach the end
                continue  # Retry with the next index
            
            # Apply transformations (convert to tensor and resize image)
            image = self.transform(image)

            # Retrieve the subCategory label and its corresponding caption
            label = sample["subCategory"].lower()
            label_idx = next(id for id, class_name in self.captions.items() 
                            if class_name == label)
            # Tokenize the caption using the tokenizer function
            cap, mask = self.tokenizer(text=self.captions[label_idx])

            mask = mask.clone().detach()
            
            if len(mask.size()) == 1:
                mask = mask.unsqueeze(0)
            
            return {
                "image": image, 
                "caption": cap, 
                "mask": mask
            }

def create_dataloader(data_path: str, 
                      image_size: tuple[int, int], 
                      test_size: float = 0.1, 
                      batch_size: int = 32):
    
    df = pd.read_csv(data_path, usecols=["id", "subCategory"])
    train_df, val_df = train_test_split(df, test_size=test_size, 
                                        random_state=42)

    train_dataset = FashionDataset(data_frame=train_df, tokenizer=tokenization, 
                                   target_size=image_size)
    val_dataset = FashionDataset(data_frame=val_df, tokenizer=tokenization, 
                                 target_size=image_size)
    test_dataset = FashionDataset(data_frame=val_df, tokenizer=tokenization, 
                                  target_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            num_workers=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            num_workers=2, shuffle=False)
    return train_loader, val_loader, test_loader


def main():
    df = pd.read_csv("data/styles.csv", usecols=["id", "subCategory"])
    dataset = FashionDataset(
        data_frame=df, 
        target_size=(80, 80), 
        tokenizer=tokenization
    )
    data = next(iter(dataset))
    print(data['image'].shape) # [C, H, W]
    print(data['caption'].shape) # [seq_len]
    print(data['mask'].shape) # [1, seq_len]

    train_loader, val_loader, test_loader = create_dataloader(
        data_path="data/styles.csv", 
        image_size=(80, 80), 
        test_size=0.1, 
        batch_size=32
    )
    data = next(iter(train_loader))
    print(data['image'].shape) # [B, C, H, W]
    print(data['caption'].shape) # [B, seq_len]
    print(data['mask'].shape) # [B, 1, seq_len]

if __name__ == "__main__":
    main()