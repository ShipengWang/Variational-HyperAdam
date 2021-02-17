import torch
from optimizee.__init__ import *
from optimizee.modules import ScaleMinus
import torch.optim as optim

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Square(Optimizee):
    def __init__(self, kwargs):
        super().__init__()
        for key_ in ['is_scaling', 'dim']:
            if key_ not in kwargs:
                raise KeyError
        self.scale = kwargs['is_scaling']
        self.dim = kwargs['dim']
        self.add_module('minus', ScaleMinus(self.dim, scale=self.scale))

    def get_train_data_loader(self, shuffle=True):
        pass

    def get_train_iter(self):
        pass

    def next_train_batch(self, device):
        return None

    def forward(self, inputs=None):
        self.output = self.minus()

    def get_loss(self, data_batch=None):
        self.forward()
        # ||w*x-v||^2 / dim
        loss = torch.mean(torch.clamp(self.output ** 2, 0.0, 10.0))
        return loss


def main():
    kwargs = {
        'is_scaling': True,
        'dim': 20
    }
    model = Square(kwargs)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.get_train_data_loader()
    for _ in range(3000):
        model.reset_parameters()
        # model.to(device_)
        model.get_train_iter()
        for epoch in range(2000):
            model.to(device_)
            for i in range(100):
                model.next_train_batch(device_)
                loss = model.get_loss()
                print(epoch, i, loss)
                model.get_grad(loss)
                optimizer.step()


if __name__ == '__main__':
    main()
