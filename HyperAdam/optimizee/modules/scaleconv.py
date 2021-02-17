import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import collections
from itertools import repeat
from . import ScaleModule


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class _ConvNd(ScaleModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias,
                 scale=None):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.scale = scale
        if self.scale is not None:
            self.r = scale
            if transposed:
                self.scale_weight = Parameter(torch.Tensor(
                    in_channels, out_channels // groups, *kernel_size), requires_grad=False)
            else:
                self.scale_weight = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size), requires_grad=False)
            if self.bias is not None:
                self.scale_bias = Parameter(torch.Tensor(out_channels), requires_grad=False)
        # self.reset_parameters()


class ScaleConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 scale=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, scale=scale)

    def forward(self, inputs):
        if self.scale is not None:
            weight, bias = self.scaling()
            return F.conv2d(inputs, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(inputs, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
