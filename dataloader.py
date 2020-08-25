import glob
import torch
from torch.utils import data

from utils.utils import read_json


class MainDataset(data.Dataset):
    def __init__(self, root, phase):
        """
        self.root = root
        self.phase = phase
        self.file_list = glob.glob(f"{root}/{phase}/*.json")
        """

    def __len__(self):
        """
        return len(self.file_list)
        """

    def __getitem__(self, idx):
        """
        file_path = self.file_list[idx]
        item = read_json(file_path)
        return item
        """
        pass


def data_loader():
    """
    dataset = MainDataset(root, phase,
                          context_max_len, context_word_len,
                          query_max_len, query_word_len)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
    """
    pass