import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataloader import data_loader
from models.bidaf_imple.model import BiDAF
from models.docqa_imple.model import DocQA

from trainer import Trainer
from evaluation import f1_score

from utils.utils import read_json


def get_model(name):
    """
    if name.lower() == "bidaf":
        return BiDAF(w_vocab_size=_w_vocab_size, w_emb_size=_w_emb_size)
    elif name.lower() == "docqa":
        return DocQA(w_vocab_size=_w_vocab_size, w_emb_size=_w_emb_size)
    else:
        raise AssertionError()
    """


def forward(data, model, loss_fn, device):
    """
    qc_ids = data['qc_ids'].to(device)

    mask = torch.zeros_like(cw_ids) != cw_ids

    logits = model(cw_ids, cc_ids, qw_ids, qc_ids)
    loss = loss_fn(start_logits, start_targets)
    return logits, loss
    """


def get_metric(logits, data):
    """
    start_logits, end_logits = logits
    start_logits, end_logits = start_logits.squeeze().cpu(), end_logits.squeeze().cpu()

    answers = data['answer']

    for i in range(n_samples):
        true_answer = answers[i]

        pred_answer = contexts[i][start_char_idx:end_char_idx+1]
        f1 += f1_score(pred_answer, true_answer)
    f1_result = 100.0 * f1 / n_samples
    return f1_result
    """


if __name__ == "__main__":
    import argparse

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _root = './data'
    _neg_inf = -1e10

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()

    config_path = args
    config = read_json(config_path)
    _lr = config.lr
    _epochs = config.epochs
    _batch_size = config.batch
    _print_iter = config.print_iter
    _model_type = config.model_type
    _save_model_dir = config.save_model_dir
    _seed = config.seed

    _run_name = f"M-{_model_type}-step_size-{_step_size}-gamma-{_gamma}-BSZ-{_batch_size}-LR-{_lr}"

    print(_run_name)
    torch.manual_seed(_seed)

    train_loader = data_loader(_root, "train",
                               _context_max_len, _context_word_len,
                               _query_max_len, _query_word_len, batch_size=_batch_size)
    dev_loader = data_loader(_root, "validate",
                             _context_max_len, _context_word_len,
                             _query_max_len, _query_word_len, batch_size=_batch_size)

    model = get_model(_model_type)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam([param for param in model.parameters() if param.requires_grad],
                     lr=_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=_step_size, gamma=_gamma)

    trainer = Trainer(train_loader, dev_loader,
                      model, loss_fn, optimizer, scheduler,
                      forward, get_metric,
                      device)

    trainer.train(_epochs, _batch_size, _print_iter, _save_model_dir, _run_name)
