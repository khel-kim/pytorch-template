from torch.optim import Adam
from models.bert import Bert
from models.albert import Albert
from models.distilbert import DistilBert
from transformers import BertTokenizer, AlbertTokenizer, DistilBertTokenizer
from modules.losses import CategoricalCrossEntropy


def get_tokenizer(name, size):
    if name == 'bert':
        return BertTokenizer.from_pretrained(f"bert-{size}-uncased")
    elif name == "albert":
        return AlbertTokenizer.from_pretrained(f"albert-{size}-v2")
    elif name == "distilbert":
        return DistilBertTokenizer.from_pretrained(f"distilbert-{size}-uncased")
    else:
        raise AssertionError()


def get_model_class(name):
    if name == "bert":
        return Bert
    elif name == "albert":
        return Albert
    elif name == "distilbert":
        return DistilBert
    else:
        raise AssertionError()


def get_loss_fn_class(name):
    if name == "cce":
        return CategoricalCrossEntropy
    else:
        raise AssertionError()


def get_optim_class(name):
    if name == "adam":
        return Adam
    else:
        raise AssertionError()


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from dataset import CustomDataset
    from modules.callbacks import ModelCheckpoint
    from modules.trainer import Trainer
    from modules.evaluations import Accuracy, Precision
    from utils.utils import read_json, set_random_seed, str2bool
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--tpu_ip_address", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config_path
    tpu_ip_address = args.tpu_ip_address

    config = read_json(config_path)
    pprint(config)
    _env, _model, _training = config["env"], config["model"], config["training"]

    _save_model_root = "saved_models"  # hard coding
    _using_time = False  # hard coding
    _root = "data/corona_nlp"  # hard coding

    # start!!
    set_random_seed(_env['seed'])

    project_name = _root.split("/")[-1]
    run_name = (f"{_model['name']}_{_model['size']}-"
                f"lr_{_training['lr']}-bsz_{_training['batch_size']}-"
                f"seed_{_env['seed']}")
    now = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')

    tokenizer = get_tokenizer(_model['name'], _model['size'])

    train_dataset = CustomDataset(_root, 'train', tokenizer, _training["max_len"])
    dev_dataset = CustomDataset(_root, 'dev', tokenizer, _training["max_len"])

    Model = get_model_class(_model['name'])
    Opt = get_optim_class(_model['opt'])
    Loss_fn = get_loss_fn_class(_model['loss'])
    model = Model(n_outputs=train_dataset.n_outputs, size=_model['size'],
                  pretrained_model_path=str2bool(_model['pretrained_model_path']))

    metric_dic = {
        "acc": Accuracy(),
        "precision": Precision()
    }
    callbacks = [
        ModelCheckpoint(f"{_save_model_root}/{run_name}.pth", monitor='dev_loss', mode="min")
    ]

    trainer = Trainer(model=model, loss_fn_class=Loss_fn, optimizer_class=Opt, metrics=metric_dic)
    trainer.fit(train_dataset, dev_dataset, lr=_training['lr'], epochs=_training['epochs'],
                batch_size=_training['batch_size'], callbacks=callbacks)
