import os
import torch
from tqdm import tqdm
from utils.utils import AverageMeter


class Trainer(object):
    def __init__(self, train_loader, dev_loader,
                       model, loss_fn, optimizer, scheduler,
                       forward, get_metric,
                       device):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.forward = forward
        self.get_metric = get_metric
        self.device = device

        self.n_batches = len(self.train_loader)

        self.model.to(device)
        self.loss_fn.to(device)

    def train(self, epochs, batch, print_iter, save_model_dir, run_name):
        train_loss = AverageMeter('train_loss')
        train_metric = AverageMeter('train_metric')
        val_loss = AverageMeter("val_loss")
        val_metric = AverageMeter("val_metric")
        print(self.model)
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in self.model.parameters())
        print("num of parameter : ", total_params)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("num of trainable_parameter : ", trainable_params)
        print("------------------------------------------------------------")
        monitor_value = 1e10
        _patience = 2
        stop_count = 0
        for epoch in range(epochs):
            print("training epoch {}...".format(epoch))
            self.model.train()
            for iter_, data in tqdm(enumerate(self.train_loader)):
                logits, loss = self.forward(data, self.model, self.loss_fn, self.device)
                metric = self.get_metric(logits, data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.update(loss, n=batch)
                train_metric.update(metric, n=batch)

                if (iter_ + 1) % print_iter == 0:
                    _epoch = epoch + ((iter_ + 1) / self.n_batches)
                    print("\t[{:.3f}/{:d}] {} {}\n".format(
                        _epoch, epochs, train_loss, train_metric))
                # break ###############
            print("\n\tevaluating validation...")
            with torch.no_grad():
                self.model.eval()
                for iter_, data in enumerate(self.dev_loader):
                    logits, loss = self.forward(data, self.model, self.loss_fn, self.device)
                    metric = self.get_metric(logits, data)
                    val_loss.update(loss, n=batch)
                    val_metric.update(metric, n=batch)

                if val_loss.avg < monitor_value:
                    self.save_model(save_model_dir, run_name)
                    monitor_value = val_loss.avg
                    stop_count = 0
                    print("\t\tmodel is saved!")
                else:
                    stop_count += 1
                    print("\t\tstop count is increased ({}/{})".format(stop_count, _patience))
                print("\t\t{}, {}\n".format(val_loss, val_metric))

            if stop_count > _patience:
                break
            train_loss.reset()
            train_metric.reset()
            val_loss.reset()
            val_metric.reset()

            self.scheduler.step()

    def save_model(self, directory, name):
        os.makedirs(directory, exist_ok=True)
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        torch.save(state, os.path.join(directory, name + ".pth"))