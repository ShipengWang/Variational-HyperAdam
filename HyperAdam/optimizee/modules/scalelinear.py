import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .module import ScaleModule


class ScaleLinear(ScaleModule):
    def __init__(self, in_features, out_features,
                 bias=True, scale=None,
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if self.scale is not None:
            self.r = scale
            self.scale_weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
            if self.bias is not None:
                self.scale_bias = Parameter(torch.Tensor(out_features), requires_grad=False)
        # self.reset_parameters()

    def forward(self, inputs):
        if self.scale is not None:
            weight, bias = self.scaling()
            return F.linear(inputs, weight, bias)
        else:
            return F.linear(inputs, self.weight, self.bias)
