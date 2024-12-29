import numpy as np
import torch 
import evaluate
import pickle
import nltk
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

# nltk.download(info_or_id="treebank")
# tagged_sentences = list(nltk.corpus.treebank.tagged_sents())  
# # Save the tagged_sentences to a file
# with open('pos_tagging_data.pickle', 'wb') as f:
#     pickle.dump(tagged_sentences, f)


class Config:
    model_name: str = "QCRI/bert-base-multilingual-cased-pos-english"
    model_to_push_hf: str = "DucPTIT/bert-base-pos-tagging"
    path_data: str = "data/pos_tagging_data.pickle"
    sequence_length: int = 500
    batch_size: int =32
    num_epochs: int = 10

model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=Config.model_name, 
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=Config.model_name
)

def prepare_data(path_data: str):
    """
    Một list các list, mỗi list chứa các tuple(token, label). 
    Mỗi list chứa các tuple tương ứng với 1 câu
    """
    with open(path_data, "rb") as f:
        pos_data = pickle.load(f) # list[list[tuple[token, label]]]

    sentences, sentence_tags = [], []
    for sample in pos_data:
        tokens, labels = zip(*sample)
        sentences.append([token.lower() for token in tokens])
        sentence_tags.append([label for label in labels])
    
    # for i in range(len(sentences)):
    #     print(sentences[i], sentence_tags[i])
    #     print("=" * 100)
    #     if i == 10:
    #         break

    train_sentences, val_sentences, train_tags, val_tags =train_test_split(
        sentences, sentence_tags, test_size=0.3
    )

    val_sentences, test_sentences, val_tags, test_tags = train_test_split(
        val_sentences, val_tags, test_size=0.5
    )
    data = {
        "train_sentences": train_sentences, 
        "val_sentences": val_sentences,
        "train_tags": train_tags, 
        "val_tags": val_tags,
        "test_sentences": test_sentences, 
        "test_tags": test_tags
    }
    return data

class PosDataset(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


def main():
    prepare_data(path_data=Config.path_data)


if __name__ == "__main__":
    main()