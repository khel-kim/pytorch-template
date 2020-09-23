import torch.nn as nn
from torch.nn import functional as F


class CategoricalCrossEntropy(nn.Module):
    def __init__(self, from_logits, reduction="mean"):
        super(CategoricalCrossEntropy, self).__init__()
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred, true):
        if self.from_logits is True:
            pred = F.log_softmax(pred, dim=-1)
        loss = -(true * pred).sum(-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum(-1)
        else:
            raise AssertionError()
