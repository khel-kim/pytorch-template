import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    print("Using torch_xla")
except ModuleNotFoundError:
    print("Not using torch_xla")


def move_to(obj, device):
    """https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to(value, device) for value in obj]
    else:
        raise AssertionError()


class AverageMeter(object):
    val, avg, sum, count = [None] * 4

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        template = "{} {:.3f} ({:.3f})"
        return template.format(self.name, self.val, self.avg)


class Trainer(object):
    def __init__(self, model, loss_fn_class, optimizer_class, metrics):
        self.using_tpu = True if "COLAB_TPU_ADDR" in os.environ or "TPU_IP_ADDRESS" in os.environ else False

        if self.using_tpu:
            self.model = xmp.MpModelWrapper(model)
        else:
            self.model = model
        self.Loss_fn = loss_fn_class
        self.Opt = optimizer_class
        self.metrics = metrics

        self.train_eval = {"train_loss": AverageMeter("train_loss")}
        for key in metrics.keys():
            self.train_eval[f"train_{key}"] = AverageMeter(f"train_{key}")

        self.dev_eval = {"dev_loss": AverageMeter("dev_loss")}
        for key in metrics.keys():
            self.dev_eval[f"dev_{key}"] = AverageMeter(f"dev_{key}")

    def fit(self, train_dataset, dev_dataset, lr, epochs, batch_size, callbacks):
        if self.using_tpu:
            xmp.spawn(self.map_fn, args=(train_dataset, dev_dataset, lr, epochs, batch_size, callbacks),
                      nprocs=8, start_method='fork')  # hard coding
        else:
            index = -1
            self.map_fn(index, train_dataset, dev_dataset, lr, epochs, batch_size, callbacks)

    def map_fn(self, index, train_dataset, dev_dataset, lr, epochs, batch_size, callbacks):
        if self.using_tpu is True:
            device = xm.xla_device()
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_loader = self.make_loader(train_dataset, batch_size, 'train')
        dev_loader = self.make_loader(dev_dataset, batch_size, 'dev')

        model = self.model.to(device)
        if self.using_tpu:
            opt = self.Opt([param for param in model.parameters() if param.requires_grad],
                           lr=lr*xm.xrt_world_size(), weight_decay=1e-4)  # hard coding
        else:
            opt = self.Opt([param for param in model.parameters() if param.requires_grad],
                           lr=lr, weight_decay=1e-4)  # hard coding

        loss_fn = self.Loss_fn(from_logits=True)

        callback_kwargs = {
            "model": model,
            "eval_dic": self.dev_eval,
        }

        for callback in callbacks:
            callback.train_init(**callback_kwargs)

        for epoch in range(epochs):
            if self.using_tpu:
                xm.rendezvous("training is starting!")
                if xm.is_master_ordinal():
                    print(f"\nepoch : {epoch+1} / {epochs}")
                now_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
            else:
                print(f"epoch : {epoch+1} / {epochs}")
                now_train_loader = train_loader
            model.train()
            for step, batch in enumerate(now_train_loader):
                logits, y, loss = self.compute_batch(model, batch, device, loss_fn, opt, phase='train')

                if self.using_tpu:
                    xm.rendezvous("update is starting!")
                    self.update(logits, y, loss, 'train', batch_size)
                    xm.rendezvous("update is ended!")
                    if xm.is_master_ordinal():
                        self.show_log(step*xm.xrt_world_size(), train_dataset, batch_size, 'train')
                else:
                    self.update(logits, y, loss, 'train', batch_size)
                    self.show_log(step, train_dataset, batch_size, 'train')

            if self.using_tpu:
                xm.rendezvous("batch is done!")
                if xm.is_master_ordinal():
                    print()
            else:
                print()

            model.eval()
            with torch.no_grad():
                if self.using_tpu:
                    now_dev_loader = pl.ParallelLoader(dev_loader, [device]).per_device_loader(device)
                else:
                    now_dev_loader = dev_loader
                for step, batch in enumerate(now_dev_loader):
                    logits, y, loss = self.compute_batch(model, batch, device, loss_fn, opt, phase='dev')

                    if self.using_tpu:
                        xm.rendezvous("update is starting!")
                        self.update(logits, y, loss, 'dev', batch_size)
                        xm.rendezvous("eval update is ended!")
                        if xm.is_master_ordinal():
                            self.show_log(step*xm.xrt_world_size(), dev_dataset, batch_size, 'dev')
                    else:
                        self.update(logits, y, loss, 'dev', batch_size)
                        self.show_log(step, dev_dataset, batch_size, 'dev')

                if self.using_tpu:
                    xm.rendezvous("batch is done!")
                    if xm.is_master_ordinal():
                        print()
                else:
                    print()
            self.on_epoch_end(callbacks)

        if self.using_tpu:
            xm.rendezvous("training is over!")

    def make_loader(self, dataset, batch_size, phase):
        shuffle = True if phase == "train" else False
        if self.using_tpu:
            sampler = DistributedSampler(dataset=dataset,
                                         num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(),
                                         shuffle=shuffle)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=xm.xrt_world_size(), drop_last=True)
        else:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def compute_batch(self, model, batch, device, loss_fn, opt, phase):
        y = batch.pop("y")
        x = batch
        x, y = move_to([x, y], device)
        logits = model(x)
        loss = loss_fn(logits, y)
        if phase == "train":
            opt.zero_grad()
            loss.backward()
            if self.using_tpu:
                xm.optimizer_step(opt)
            else:
                opt.step()
        return logits, y, loss

    def update(self, logits, y, loss, phase, batch_size):
        eval_dic = self.train_eval if phase == "train" else self.dev_eval
        eval_dic[f"{phase}_loss"].update(float(loss), batch_size)
        for name, metric in self.metrics.items():
            eval_dic[f"{phase}_{name}"].update(metric.compute(logits.to('cpu'), y.detach().to('cpu')), batch_size)

    def show_log(self, step, dataset, batch_size, phase):
        eval_dic = self.train_eval if phase == "train" else self.dev_eval
        total_data = len(dataset)

        rows, columns = os.popen("stty size", 'r').read().split()
        columns = int(columns)
        bar_width = columns - 90

        n_data = step * batch_size
        n_bar = int((n_data / total_data) * bar_width)

        template = "{} : {}/{} |{}{}| "
        for key in range(len(eval_dic)):
            template += "{}, "
        template = template[:-2]
        log = template.format(
            phase,
            n_data, total_data,
            "*" * n_bar, "-" * (bar_width - n_bar),
            *sorted([f"{key}:{'{:.3f}'.format(value.avg)}" for key, value in eval_dic.items()]))

        log += f'{" " * (columns - len(log))}'
        sys.stdout.write("\r" + log)
        sys.stdout.flush()

    def on_epoch_end(self, callbacks):
        def execute():
            for callback in callbacks:
                callback.on_epoch_end()

            # reset
            for name, metric in self.train_eval.items():
                self.train_eval[name].reset()
            for name, metric in self.dev_eval.items():
                self.dev_eval[name].reset()

        if self.using_tpu:
            xm.rendezvous("train is done!")
            if xm.is_master_ordinal():
                execute()
                xm.rendezvous("on_epoch_end is done!")
            else:
                xm.rendezvous("on_epoch_end is done!")
        else:
            execute()
