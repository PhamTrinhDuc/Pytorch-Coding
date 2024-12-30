import os
import torch
import evaluate
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer


class Config:
    model_name: str = "d4data/biomedical-ner-all"
    model_push_hf: str = "DucPTIT/bert-ner-medical"
    dataset_dir: str = "data/ner_data"
    max_len: int = 512
    per_device_batch_size: int = 16
    num_epochs: int = 20
    device = 'cuda' if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=Config.model_name
)
model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=Config.model_name, 
    label2id = Config.label2id, 
    id2label = Config.id2label, 
    ignore_mismatched_sizes=True
)

class PreprocessingMaccrobat:
    def __init__(self, dataset_folder, tokenizer):
        self.file_ids = [f.split(".")[0] for f in os.listdir(dataset_folder) if f.endswith('.txt')]

        self.text_files = [f+".txt" for f in self.file_ids]
        self.anno_files = [f+".ann" for f in self.file_ids]

        self.num_samples = len(self.file_ids)

        self.texts: List[str] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.text_files[i])
            with open(file_path, 'r') as f:
                self.texts.append(f.read())

        self.tags: List[Dict[str, str]] = []
        for i in range(self.num_samples):
            file_path = os.path.join(dataset_folder, self.anno_files[i])
            with open(file_path, 'r') as f:
                text_bound_ann = [t.split("\t") for t in f.read().split("\n") if t.startswith("T")]
                text_bound_lst = []
                for text_b in text_bound_ann:
                    label = text_b[1].split(" ")
                    try:
                        #kiểm tra xem 2 index có đúng là kiểu số nguyên không
                        _ = int(label[1])
                        _ = int(label[2])
                        tag = {
                            "text": text_b[-1],
                            "label": label[0],
                            "start": label[1],
                            "end": label[2]
                        }
                        text_bound_lst.append(tag)
                    except:
                        pass

                self.tags.append(text_bound_lst)
        self.tokenizer = tokenizer

    def process(self) -> Tuple[List[List[str]], List[List[str]]]:
        input_texts = []
        input_labels = []

        for idx in range(self.num_samples):
            full_text = self.texts[idx]
            tags = self.tags[idx]
            # 4 => 6 có thực thể; 0, 1, 2, 3 --- 7, 8 không có thực thể


            label_offset = []
            continuous_label_offset = []
            for tag in tags:
                offset = list(range(int(tag["start"]), int(tag["end"])+1))
                label_offset.append(offset)
                continuous_label_offset.extend(offset)

            all_offset = list(range(len(full_text)))
            zero_offset = [offset for offset in all_offset if offset not in continuous_label_offset]
            zero_offset = self.find_continuous_ranges(zero_offset)

            self.tokens = []
            self.labels = []
            self._merge_offset(full_text, tags, zero_offset, label_offset)
            assert len(self.tokens) == len(self.labels), f"Length of tokens and labels are not equal"

            input_texts.append(self.tokens)
            input_labels.append(self.labels)

        return input_texts, input_labels

    def _merge_offset(self, full_text, tags, zero_offset, label_offset):
        #zero_offset = [[0, 1, 2, 3], [7, 8], [9, 10, 11]]
        #label_offset = [4, 5, 6]

        i = j = 0
        while i < len(zero_offset) and j < len(label_offset):
            if zero_offset[i][0] < label_offset[j][0]:
                self._add_zero(full_text, zero_offset, i)
                i += 1
            else:
                self._add_label(full_text, label_offset, j, tags)
                j += 1

        while i < len(zero_offset):
            self._add_zero(full_text, zero_offset, i)
            i += 1

        while j < len(label_offset):
            self._add_label(full_text, label_offset, j, tags)
            j += 1

    def _add_zero(self, full_text, offset, index):
        start, *_ ,end =  offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0]+1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)

        self.tokens.extend(text_tokens)
        self.labels.extend(
            ["O"]*len(text_tokens)
        )

    def _add_label(self, full_text, offset, index, tags):
        start, *_ ,end =  offset[index] if len(offset[index]) > 1 else (offset[index][0], offset[index][0]+1)
        text = full_text[start:end]
        text_tokens = self.tokenizer.tokenize(text)

        self.tokens.extend(text_tokens)
        self.labels.extend(
            [f"B-{tags[index]['label']}"] + [f"I-{tags[index]['label']}"]*(len(text_tokens)-1)
        )

    @staticmethod
    def build_label2id(tokens: List[List[str]]):
        label2id = {}
        id_counter = 0
        for token in [token for sublist in tokens for token in sublist]:
            if token not in label2id:
                label2id[token] = id_counter
                id_counter += 1
        return label2id

    @staticmethod
    def find_continuous_ranges(data: List[int]): # 0, 1, 2, 3, 7, 8
        if not data:
            return []
        ranges = []
        start = data[0]
        prev = data[0]
        for number in data[1:]:
            if number != prev + 1:
                ranges.append(list(range(start, prev + 1)))
                start = number
            prev = number
        ranges.append(list(range(start, prev + 1)))
        return ranges #[[0, 1, 2, 3], [7, 8] ....]


class NERDataset(Dataset):
    def __init__(self, input_texts: list, input_labels: list, 
                 tokenizer: AutoTokenizer, label2id: dict, max_len: int):
        self.tokens = input_texts
        self.labels = input_labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        tokens = self.tokens[index]
        labels = self.labels[index]

        input_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_tokens)

        input_ids = self.pad_and_truncate(inputs=input_tokens, pad_id = self.tokenizer.pad_token_ids)
        attention_mask = self.pad_and_truncate(inputs=attention_mask, pad_id=0)
        labels = self.pad_and_truncate(inputs=labels, pad_id=0)

        return {
            "input_ids": torch.as_tensor(input_ids),
            "attention_mask": torch.as_tensor(attention_mask),
            "labels": torch.as_tensor(labels)
        }

    def pad_and_truncate(self, inputs: List[int], pad_id: int):
        if len(inputs) < self.max_len:
            padded_inputs = inputs + [pad_id] * (self.max_len - len(inputs))
        else:
            padded_inputs = inputs[:self.max_len]
        return padded_inputs
    

def prepare_data(folder_dir: str):
    processor = PreprocessingMaccrobat(dataset_folder=folder_dir, tokenizer=tokenizer)
    input_texts, input_labels = processor.process()
    Config.label2id = processor.build_label2id(tokens=input_texts)
    Config.id2label = {id: label for label, id in Config.label2id.items()}


    inputs_train, inputs_val, labels_train, labels_val = train_test_split(
        input_texts, input_labels, test_size=0.2, random_state=42
    ) 
    train_dataset = NERDataset(input_texts=inputs_train, input_labels=labels_train, 
                               tokenizer=tokenizer, label2id=Config.label2id)
    val_dataset = NERDataset(input_texts=inputs_val, input_labels=labels_val, 
                             tokenizer=tokenizer, label2id=Config.label2id)
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")

    predictions, labels = eval_pred
    mask = labels != 0
    predictions = np.argmax(predictions, axis = -1)
    return metric.compute(predictions=predictions[mask], references=labels[mask])


def trainning(train_dataset, val_dataset):
    args = TrainingArguments(
        output_dir="./NER/checkpoint", 
        learning_rate=1e-4, 
        per_device_eval_batch_size=Config.per_device_batch_size, 
        per_device_train_batch_size=Config.per_device_batch_size, 
        eval_strategy="epoch", 
        save_strategy="epoch", 
        num_train_epochs=Config.num_epochs, 
        optim="adamw_torch", 
        load_best_model_at_end=True
    )

    trainer = Trainer(
        args=args, 
        model=model, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.push_to_hub(Config.model_push_hf)
    tokenizer.push_to_hub(Config.model_push_hf)


def inference(text: str):
    inputs = torch.as_tensor(tokenizer.convert_tokens_to_ids(text.split())).to(Config.device)
    output = model(inputs)
    # print(output.logits.shape) # [1, 63, 83]

    _, preds = torch.max(output.logits, -1)
    preds = preds[0].cpu().numpy()

    for token, pred in zip(text.split(), preds):
        print(f"{token}\t{Config.id2label[pred]}")


def main():
    train_dataset, val_dataset = prepare_data(folder_dir=Config.dataset_dir)
    trainning(train_dataset=train_dataset, val_dataset=val_dataset)

    test_sentence = """A 48 year - old female presented with vaginal bleeding and abnormal Pap smears .
    Upon diagnosis of invasive non - keratinizing SCC of the cervix ,
    she underwent a radical hysterectomy with salpingo - oophorectomy
    which demonstrated positive spread to the pelvic lymph nodes and the parametrium .
    Pathological examination revealed that the tumour also extensively involved the lower uterine segment .
    """
    inference(text=test_sentence)



if __name__ == "__main__":
    main()