import torch.nn as nn
from transformers import DistilBertModel


class DistilBert(nn.Module):
    def __init__(self, n_outputs, size, pretrained_model_path=False):
        super(DistilBert, self).__init__()
        self.n_outputs = n_outputs
        self.size = size
        self.pretrained_model_path = pretrained_model_path

        if self.pretrained_model_path is False:
            self.huggingface_model = DistilBertModel.from_pretrained(f"distilbert-{size}-uncased")
        else:
            self.huggingface_model = DistilBertModel.from_pretrained(pretrained_model_path)
        self.dropout = nn.Dropout(0.1)  # hard coding
        self.out_proj = nn.Linear(self.huggingface_model.config.hidden_size, n_outputs)

    def forward(self, x):
        x = self.huggingface_model(**x)
        x = x[0][:, 0, :]  # cls token id : 0
        x = self.dropout(x)
        out = self.out_proj(x)
        return out
