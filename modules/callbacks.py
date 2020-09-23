import torch
INF = 10e10


class ModelCheckpoint(object):
    def __init__(self, filepath, monitor, mode,
                 save_best_only=True, save_weight_only=False, save_freq='epoch'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weight_only = save_weight_only
        self.save_freq = save_freq

        self.model = None
        self.eval_dic = None

        if mode == "max":
            self.value = -INF
        elif mode == "min":
            self.value = INF
        else:
            raise AssertionError()

    def train_init(self, model, eval_dic, **kwargs):
        self.model = model
        self.eval_dic = eval_dic

    def on_epoch_end(self):
        if self.mode == "max":
            if self.value < self.eval_dic[self.monitor].avg:
                torch.save(self.model.state_dict(), self.filepath)
                self.value = self.eval_dic[self.monitor].avg
                print("model is saved!")
        elif self.mode == "min":
            if self.value > self.eval_dic[self.monitor].avg:
                torch.save(self.model.state_dict(), self.filepath)
                self.value = self.eval_dic[self.monitor].avg
                print("model is saved!")
        else:
            raise AssertionError()
