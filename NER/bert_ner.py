import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification



class PreprocessingMaccrobat:
    def __init__(self, folder_dir: str, tokenizer: AutoTokenizer):
        
        # get dir ids of files 
        list_dir_ids = [os.path.join(folder_dir, path.split(".")[0]) for path in os.listdir(folder_dir) if path.endswith(".txt")] 
        
        # get dir of text file and label file
        list_dir_texts = [path_id + ".txt" for path_id in list_dir_ids]
        list_dir_tags = [path_id + ".ann" for path_id in list_dir_ids]
        self.num_samples = len(list_dir_ids)

        # get texts
        self.texts: List[str] = []
        for path_text in list_dir_texts:
            with open(path_text, 'r') as f:
                self.texts.append(f.read()) 

        # get labels
        self.tags: List[List[Dict[str, str]]] = [] # store all files 
        for path_tag in list_dir_tags:
            with open(path_tag, 'r') as f:
                text_bound_ann = [line.split("\t") for line in f.read().split("\n") if line.startswith("T")] # 1 line: "T1	Age 18 27	34-yr-old"
                text_bound_lst = [] # store one file
                for item in text_bound_ann:
                    # item: ["T1", "Age 18 27", "34-yr-old"]
                    # check start and end is integer
                    label = item[1].split(" ")
                    try:
                        _ = int(label[1])
                        _ = int(label[2])

                        data = {
                            "text": item[-1],
                            "label": label[0], 
                            "start": label[1],
                            "end": label[2]
                        }
                        text_bound_lst.append(data)
                    except Exception as e:
                        print(f'Start: {label[1]} and End: {label[2]} are not integer')
                self.tags.append(text_bound_lst)
        self.tokenizer = tokenizer


    def process(self) -> Tuple[List[List[str]], Tuple[List[List[str]]]]:
        input_texts = []
        input_labels = []
        for i in range(self.num_samples):
            text: str = self.texts[i]
            tags: list[dict] = self.tags[i]

            # 4 => 6 có thực thể; 0, 1, 2, 3 --- 7, 8 không có thực thể
            labels_offset = []
            labels_continuous_offset = []
            for tag in tags:
                offset: list[int] = list(range(tag['start'], tag["end"] + 1)) # [4, 5, 6]
                labels_offset.append(offset) # [[4, 5, 6]]
                labels_continuous_offset.extend(offset) # [4, 5, 6]

            all_offset: list[int] =  list(range(len(text))) # [0, 1, 2, 3, 4, 5, 6, 7, 8]
            zero_offset = [offset for offset in all_offset if offset not in labels_continuous_offset] # [0, 1, 2, 3, 7, 8]  
            zero_offset = self.find_contiuous_offset(zero_offset) # [[0, 1, 2, 3], [7, 8]]
        
        

def main():
    processor = PreprocessingMaccrobat(folder_dir="data/ner_data")
    print(len(processor.texts))
    print(len(processor.tags))
    


if __name__ == "__main__":
    main()