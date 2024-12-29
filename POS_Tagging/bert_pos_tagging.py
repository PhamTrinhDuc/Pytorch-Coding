import numpy as np
import torch 
import evaluate
import pickle
import nltk
from typing import List, Tuple
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

# nltk.download(info_or_id="treebank")
# tagged_sentences = list(nltk.corpus.treebank.tagged_sents())  
# # Save the tagged_sentences to a file
# with open('pos_tagging_data.pickle', 'wb') as f:
#     pickle.dump(tagged_sentences, f)

class Config:
    model_name: str = "QCRI/bert-base-multilingual-cased-pos-english"
    model_to_push_hf: str = "DucPTIT/bert-base-pos-tagging"
    path_data: str = "data/pos_tagging_data.pickle"
    output_dir: str = "./POS_Tagging/checkpoint"
    sequence_length: int = 500
    batch_size: int =32
    per_device_batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=Config.model_name, 
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=Config.model_name
)

# Trong data có label "-NONE-" không có trong tập label của model. 
# Defaultdict sẽ chuyển id của label "-NONE-" có id là 0 tương với token "UNK"

Config.label2id = defaultdict(int, model.config.label2id)
Config.id2label = {id: label for label, id in Config.label2id.items()}


class PosTaggingDataset(Dataset):
    def __init__(self, 
                 sentences: List[str], 
                 tags: List[str],
                 tokenizer: AutoTokenizer, 
                 label2id: dict = None, 
                 max_len: int = None):
        super().__init__()
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len or max([len(sentence) for sentence in self.sentences])
        self.label2id = label2id or Config.label2id
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index) -> dict:
        sentence_token = self.sentences[index]
        label_token = self.tags[index]
        
        input_token = self.tokenizer.convert_tokens_to_ids(sentence_token)
        attention_mask = [1] * len(input_token)
        labels = [self.label2id[token] for token in label_token]

        return {
            "input_ids": self.pad_and_truncate(inputs=input_token, pad_id=self.tokenizer.pad_token_id),
            "labels": self.pad_and_truncate(inputs=labels, pad_id=self.label2id["0"]), # token pad has label "0" 
            "attention_mask": self.pad_and_truncate(inputs=attention_mask, pad_id=0)
        }
    
    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return torch.as_tensor(padded_inputs)


def prepare_data(path_data: str) -> dict:
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
    

    train_sentences, val_sentences, train_tags, val_tags =train_test_split(
        sentences, sentence_tags, test_size=0.3
    )

    val_sentences, test_sentences, val_tags, test_tags = train_test_split(
        val_sentences, val_tags, test_size=0.5
    )

    train_dataset = PosTaggingDataset(sentences=train_sentences, tags=train_tags, tokenizer=tokenizer, label2id=Config.label2id)
    val_dataset = PosTaggingDataset(sentences=val_sentences, tags=val_tags, tokenizer=tokenizer, label2id=Config.label2id)
    test_dataset = PosTaggingDataset(sentences=test_sentences, tags=test_tags, tokenizer=tokenizer, label2id=Config.label2id)

    data = {
        "train_dataset": train_dataset, 
        "val_dataset": val_dataset, 
        "test_dataset": test_dataset,
    }

    return data


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """
    Calculate the model's accuracy based on predictions and labels,
    ignore labels marked as ignore_label.

    Parameters
    ----------
    eval_pred : Tuple[ndarray, ndarray]
        Tuple consists of 2 numpy arrays:
            - predictions: 2D array containing logits with shape (n_samples, n_classes)
            - labels: 1D array containing actual labels with shape (n_samples,)
    
    Returns
    -------
    dict[str, float]
        Dictionary contains accuracy results

    """

    accuracy = evaluate.load("accuracy")
    ignore_label = len(Config.label2id) # 46
    
    predictions, labels = eval_pred
    # Create a mask to filter labels that should not ignore_label
    mask = labels != ignore_label
    
    # Convert logits to predicted classes using argmax
    predictions = np.argmax(predictions, axis=-1)
    
    # Calculate accuracy only on non-ignored samples
    return accuracy.compute(
        predictions=predictions[mask], 
        references=labels[mask]
    )


def training(train_dataset: Dataset, val_dataset: Dataset):

    args = TrainingArguments(
        output_dir=Config.output_dir, 
        learning_rate=Config.learning_rate, 
        num_train_epochs=Config.num_epochs, 
        per_device_train_batch_size=Config.per_device_batch_size, 
        per_device_eval_batch_size= Config.per_device_batch_size, 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        report_to=None, 
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.push_to_hub(Config.model_to_push_hf)
    tokenizer.push_to_hub(Config.model_to_push_hf)
    return trainer


def inference(text: str, model, tokenizer):
    input = torch.as_tensor(tokenizer.convert_tokens_to_ids(tokens=text.split()))
    input.to(Config.device)
    outputs = model(input) # [1, 8, len(text.split())]
    _, preds = torch.max(outputs.logits, -1)
    preds = preds[0].cpu().numpy()
    pred_tags = ""
    for pred in preds:
        pred_tags += Config.id2label[pred] + " "
    
    return pred_tags
    


def main():
    # ------------- fetch data
    data = prepare_data(path_data=Config.path_data)

    # ------------- training
    trainer = training(train_dataset=data['train_dataset'], val_dataset=data['val_dataset'])

    # ------------- testing
    results = trainer.evaluate(eval_dataset=data['test_dataset'])
    print(results)

    # -------------- inference 
    model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=Config.model_to_push_hf, 
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=Config.model_to_push_hf
    )   

    text = "We are exploring the topic of deep learning"
    inference(text=text, model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    main()