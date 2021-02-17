import numpy as np
import torch.nn.functional as F
from optimizee.modules import ScaleModule


class Optimizee(ScaleModule):
    def __init__(self):
        super().__init__()
        self.train_iter = None
        self.activationFunc = None
        self.train_loader = None
        self.test_loader = None
        self.train_images = None
        self.train_labels = None
        self.output = None
        # self.loss = None

    def _get_activation(self, activation):
        def activation_func(x):
            if activation == 'sigmoid':
                return F.sigmoid(x)
            elif activation == 'relu':
                return F.relu(x)
            elif activation == 'tanh':
                return F.tanh(x)
            elif activation == 'elu':
                return F.elu(x)

        self.activationFunc = activation_func

    def get_dim(self):
        dim = 0
        for param in self.parameters():
            if param.requires_grad:
                dim += np.prod(param.shape)
        return dim

    def cycling_data(self):
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        self.train_loader = cycle(self.train_loader)

    def get_train_iter(self):
        self.train_iter = iter(self.train_loader)

    def next_train_batch(self, device):
        train_images, train_labels = next(self.train_iter)
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        return train_images, train_labels

    def reset_parameters(self):
        for module_ in self.children():
            if hasattr(module_, 'scale'):
                if module_.scale:
                    module_.reset_scale()
            module_.reset_parameters()

    def forward(self, inputs=None):
        raise NotImplementedError

    def get_loss(self, data_batch=None):
        raise NotImplementedError

    def get_grad(self, loss):
        self.zero_grad()
        loss.backward()


from . import scaleMLP
from . import cnn
from . import trivial
from . import trick
from . import sinlstmmodel

__all__ = ['Optimizee', 'trivial', 'trick',
           'scaleMLP', 'cnn', 'sinlstmmodel']
