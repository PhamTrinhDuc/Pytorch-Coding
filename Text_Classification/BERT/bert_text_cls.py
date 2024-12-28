import pandas as pd
import string
import re
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

class Config: 
    MODEL_NAME: str = "bert-base-uncased"
    max_seq_len: int = 128
    num_classes: int = 2
    batch_size: int = 128
    lr: float = 2e-5
    per_device_batch_size: int = 128
    epochs: int = 1
    path_model: str = "../BERT/checkpoint"
    hf_model: str = "DucPTIT/Bert-Classifier-Text"

class TextClassifierBERT:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = init_tokenizer()
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = self._init_model()
        self.trainer = self.init_trainer()

    def _init_model(self):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=Config.MODEL_NAME,
            num_labels = Config.num_classes,
            finetunig_task = "text-classification"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=Config.MODEL_NAME,
            config = config
        )
        # print(model)
        return model
    
    def compute_metrics(self, eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
        """
        Calculate evaluation metrics for text classification model.
        
        Args:
            eval_pred (tuple): Tuple of (predictions, labels) where:
                - predictions (np.ndarray): Model predictions/logits with shape (batch_size, num_classes)
                - labels (np.ndarray): True labels with shape (batch_size,)
        
        Returns:
            dict: Dictionary containing metric names and values:
                - accuracy (float): Proportion of correct predictions
                - f1 (float): F1 score with weighted average
                - (precision, recall, roc_auc...)
                
        Example:
            >>> pred = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])  # (2 samples, 3 classes) 
            >>> labels = np.array([1, 1])  # True labels
            >>> compute_metrics((pred, labels))
            {'accuracy': 1.0, 'f1': 1.0}
        """
        
        accuracy_metric = evaluate.load("accuracy")
        # f1_metric = evaluate.load("f1")

        predictions, labels = eval_pred
        position_pred = np.argmax(predictions, axis=1)
        results = {
            **accuracy_metric.compute(predictions=position_pred, references=labels),
            # **f1_metric.compute(predictions=position_pred, references=labels, average='weighted')
        }
        return results

    def init_trainer(self):
        args_train = TrainingArguments(
            output_dir=Config.path_model,
            per_device_eval_batch_size=Config.per_device_batch_size,
            per_device_train_batch_size=Config.per_device_batch_size,
            learning_rate=Config.lr,
            num_train_epochs=Config.epochs,
            report_to=None, # if wandb
            eval_strategy="epoch", # eval after each epoch,
            save_strategy="epoch", # save model each epoch,
            load_best_model_at_end=True, # load best model 
            # metric_for_best_model="f1", # Save the best model based on a specific metric
            # greater_is_better=True # Select greater than as True if the number is higher the better
        )

        trainer = Trainer(
            model=self.model,
            args=args_train,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator,
        )
        return trainer
    
def init_tokenizer(prrint_info: bool = False):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=Config.MODEL_NAME,
            use_fast = True
        )
        if prrint_info:
            print("Max seq length of tokenizer:", tokenizer.model_max_length)
            print(f"Token_pad: {tokenizer.pad_token}\t Id token pad: {tokenizer.pad_token_id}: ")
            print(f"All tokens special: {tokenizer.all_special_ids}\t Id all tokens special: {tokenizer.all_special_ids}")
            print("Try with example has two setences: ")
            examples = [
                "Tôi đang đi học",
                "Ngày hôm nay đẹp quá, tôi muốn ra ngoài chơi"
            ]

            results = tokenizer(
                text=examples,
                padding="max_length",
                max_length=Config.max_seq_len,
                truncation=True
            )
            print("Key results (dict): ", results.keys())
            print("Shape two senteces tokenized: ", np.array(results['input_ids']).shape)
        return tokenizer


def prepare_data(df_train: pd.DataFrame, 
                 df_val: pd.DataFrame, 
                 df_test: pd.DataFrame = None) :
    
    def _preprocess_text(text: str) -> str:
        # remove URLs https://www.
        url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
        text = url_pattern.sub(r" ", text)

        # remove HTML Tags: <>
        html_pattern = re.compile(r'<[^<>]+>')
        text = html_pattern.sub(" ", text)

        # remove puncs and digits
        replace_chars = list(string.punctuation + string.digits)
        for char in replace_chars:
            text = text.replace(char, " ")

        # remove emoji
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r" ", text)

        # normalize whitespace
        text = " ".join(text.split())

        # lowercasing
        text = text.lower()
        return text

    def _process_func(examples: list):
        tokenizer = init_tokenizer()
        results = tokenizer(
            examples['sentence'],
            padding="max_length",
            max_length = Config.max_seq_len,
            truncation = True
        ) 
        results['labels'] = examples['label']
        return results
    
    df_train['sentence'].apply(_preprocess_text)
    df_val['sentence'].apply(_preprocess_text)
    df_test['sentence'].apply(_preprocess_text)
    
    if df_test is not None:
        raw_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(df=df_train),
                "valid": Dataset.from_pandas(df=df_val),
                "test": Dataset.from_pandas(df=df_test)
            }
        )
    else:
        raw_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(df=df_train),
                "valid": Dataset.from_pandas(df=df_val),
            }
        )

    preprocessed_data = raw_dataset.map(
        function=_process_func,
        batched=True,
        desc="Running toknizer on dataset"
    )
    print(preprocessed_data)
    return preprocessed_data


def inference(text: str):
    model_name=Config.hf_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")

    id2label = {1: "POSITIVE", 0: "NEGATIVE"}
    label2id = {"POSITIVE": 1, "NEGATIVE": 0}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.forward(**inputs)

    predicted_label_index = outputs.logits.argmax(-1).item()
    predicted_label = id2label[predicted_label_index]
    print(f"The predicted label for the text is: {predicted_label}")

def main():

    # processing data
    df_train = pd.read_csv("data/cls_text/train.csv")
    df_val = pd.read_csv("data/cls_text/val.csv")
    df_test = pd.read_csv("data/cls_text/test.csv")
    preprocessed_data = prepare_data(df_train=df_train,
                                df_val=df_val,
                                df_test=df_test)
    # init instance
    classifier = TextClassifierBERT(
        train_dataset=preprocessed_data['train'],

        val_dataset=preprocessed_data['valid']
    )
    # training 
    classifier.trainer.train()

    # save model and tokenizer to hf
    classifier.model.push_to_hub(Config.hf_model)
    classifier.tokenizer.push_to_hub(Config.hf_model)
    
    
    # evaluate with test dataset 
    results = classifier.trainer.evaluate(test_dataset=preprocessed_data['test'])
    print("Results after evaluate: ")
    print(results)    


if __name__ == "__main__":
    main()