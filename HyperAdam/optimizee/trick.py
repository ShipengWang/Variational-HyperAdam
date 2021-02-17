from optimizee.__init__ import *
import torch

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mixed(Optimizee):
    def __init__(self, opt_dict):
        super().__init__()
        self.coe = {}
        for key_, value_ in opt_dict.items():
            self.add_module(key_, value_['learner'](value_['kwargs']))
            self.coe.update({key_: value_['coe']})

    # will be removed
    def children(self):
        for name, model_ in self.named_children():
            for moule_ in model_.children():
                yield moule_

    def reset_parameters(self):
        for name, model_ in self.named_children():
            model_.reset_parameters()

    def set_device(self, device):
        for name, model_ in self.named_children():
            model_.set_device(device)

    def get_train_data_loader(self, shuffle=True):
        for name, model_ in self.named_children():
            model_.get_train_data_loader(shuffle=shuffle)

    def get_train_iter(self):
        for name, model_ in self.named_children():
            model_.get_train_iter()

    def next_train_batch(self, device):
        data_batch = {}
        for name, model_ in self.named_children():
            data = model_.next_train_batch(device)
            data_batch.update({name: data})
        return data_batch

    def forward(self, inputs=None):
        for name, model_ in self.named_children():
            model_.forward()

    def get_loss(self, data_batch=None):
        loss = sum(self.coe[name] * model_.get_loss(data_batch[name]) for name, model_ in self.named_children())
        # loss = None
        # for name, model_ in self.named_children():
        #     if loss is None:
        #         loss = model_.get_loss(data_batch[name])
        #     else:
        #         loss += model_.get_loss(data_batch[name])
        return loss

    def get_losses(self, data_batch=None):
        losses = {}
        for name, model_ in self.named_children():
            losses.update({name: model_.get_loss(data_batch[name])})
        return losses


def main():
    opt_dict = {
        'MLP_Scale': {'learner': scaleMLP.ScaleMultilayerPerceptron,
                      'kwargs': {
                          'depth': 1,
                          'activation': 'sigmoid',
                          'is_scaling': True,
                          'data_path': '../../dataset/mnist',
                          'batch_size': 128
                      },  # number of hidden layers, activation, is_scaling
                      },
        'convex': {'learner': trivial.Square,
                   'kwargs': {
                       'is_scaling': True,
                       'dim': 20
                   }
                   },
    }

    model = Mixed(opt_dict)
    # module = list(model.children())
    # sub_model = list(model.named_children())
    print(model)
    model.get_train_data_loader()
    model.get_train_iter()
    model.reset_parameters()
    model.next_train_batch(device=device_)
    model.to(device=device_)
    loss = model.get_loss()
    model.get_grad(loss)
    print(model)


if __name__ == "__main__":
    main()
