import json
import pandas as pd
import numpy as np
import random
import torch


def read_txt(path):
    with open(path) as f:
        result = f.readlines()
    return result


def read_tsv(path, header="infer"):
    return pd.read_csv(path, sep='\t', header=header)


def read_csv(path, header="infer", encoding='utf-8'):
    return pd.read_csv(path, header=header, encoding=encoding)


def read_json(path):
    with open(path) as j:
        result = json.load(j)
    return result


def read_npy(path):
    return np.load(path)


def read_jsonl(path):
    return pd.read_json(path, lines=True)


def save_json(path, obj):
    with open(path, 'w') as j:
        json.dump(obj, j, ensure_ascii=False)


def save_npy(path, obj):
    path = path.replace(".npy", "")
    np.save(path, obj)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(string):
    string = string.lower()
    return True if string in ["y", "t", "true", "yes"] else False


if __name__ == "__main__":
    set_random_seed(3)
