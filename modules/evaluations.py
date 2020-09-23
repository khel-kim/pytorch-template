import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Accuracy(object):
    @staticmethod
    def compute(pred, true):
        pred_classes = torch.argmax(pred, dim=1)
        true_classes = torch.argmax(true, dim=1)
        return accuracy_score(true_classes, pred_classes)


class Precision(object):
    @staticmethod
    def compute(pred, true):
        pred_classes = torch.argmax(pred, dim=1)
        true_classes = torch.argmax(true, dim=1)
        return precision_score(true_classes, pred_classes, average='macro', zero_division=0)  # hard coding


