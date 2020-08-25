import json
import pandas as pd
import numpy as np
import random
import torch


def read_tsv(path, header="infer"):
    return pd.read_csv(path, sep='\t', header=header)


def read_csv(path, header="infer"):
    return pd.read_csv(path, header=header)


def read_json(path):
    with open(path) as j:
        result = json.load(j)
    return result


def save_json(path, obj):
    with open(path, 'w') as j:
        json.dump(obj, j, ensure_ascii=False)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"set seed as {seed}")


class AverageMeter(object):
    val, avg, sum, count = [None] * 4

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return "{} {:.3f} ({:.3f})".format(self.name, self.val, self.avg)


if __name__ == "__main__":
    set_random_seed(3)
