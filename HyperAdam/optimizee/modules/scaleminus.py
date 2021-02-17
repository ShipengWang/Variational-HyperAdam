import torch
from torch.nn.parameter import Parameter
import numpy as np
from .module import ScaleModule


class ScaleMinus(ScaleModule):
    def __init__(self, dim, scale=None):
        super().__init__()
        self.dim = dim
        self.weight = Parameter(torch.Tensor(dim))
        self.scale = scale
        self.bias_ = Parameter(torch.Tensor(dim), requires_grad=False)
        if self.scale is not None:
            self.r = scale
            self.scale_weight = Parameter(torch.Tensor(dim), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(0.0)
        self.bias_.data.uniform_(-1.0, 1.0)
        if self.scale is not None:
            self.weight.data.div_(self.scale_weight.data)

    def forward(self):
        if self.scale is not None:
            weight = self.scaling()
            return weight - self.bias_
        else:
            return self.weight - self.bias_

    def extra_repr(self):
        return 'dimensions={}'.format(
            self.dim is not None
        )
