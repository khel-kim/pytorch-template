import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import read_json


class CustomDataset(Dataset):
    def __init__(self, root, phase, tokenizer, max_len):
        self.root = root
        self.phase = phase
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cat2idx = read_json(f"{root}/info/cat2idx.json")
        self.idx2cat = {idx: cat for idx, cat in self.cat2idx.items()}
        self.n_outputs = len(self.cat2idx)

        data = read_json(f"{root}/{phase}.json")
        self.texts = data['texts']
        self.categories = data['categories'] if phase != "test" else None

        self.pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        X = self.tokenizer(text)
        X = {key: torch.LongTensor(self.pad(value)) for key, value in X.items()}
        if self.phase != "test":
            y = self.categories[idx]
            y = self.idx2onehot(y)
            X["y"] = torch.LongTensor(y)
        return X

    def pad(self, arr):
        return arr[:self.max_len] + [self.pad_token_id] * (self.max_len - len(arr))

    def idx2onehot(self, y):
        onehot = np.zeros(self.n_outputs)
        onehot[y] = 1
        return onehot


if __name__ == "__main__":
    from transformers import BertTokenizer
    from pprint import pprint
    _root = "data/corona_nlp"
    _phases = ["train", "test", "dev"]
    _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    _max_len = 25
    for _phase in _phases:
        dataset = CustomDataset(_root, _phase, _tokenizer, _max_len)

        for res in dataset:
            pprint(res)
            print(res['y'].size())
            break

