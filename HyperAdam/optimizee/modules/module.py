import torch.nn as nn


class ScaleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_weight = None
        self.scale_bias = None
        self.weight = None
        self.scale = None
        self.bias = None
        # self._tensors = OrderedDict()

    def reset_scale(self):
        self.scale_weight.data.uniform_(-self.r, self.r)
        self.scale_weight.exp_()
        if hasattr(self, 'bias'):
            if self.bias is not None:
                self.scale_bias.data.uniform_(-self.r, self.r)
                self.scale_bias.exp_()

    # def set_scale_device(self, device):
    #     self.scale_weight = self.scale_weight.to(device)
    #     if hasattr(self, 'bias'):
    #         if self.scale_bias is not None:
    #             self.scale_bias = self.scale_bias.to(device)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        if hasattr(self, 'bias'):
            if self.bias is not None:
                self.bias.data.normal_(0, 0.1)
        if self.scale is not None:
            self.weight.data.div_(self.scale_weight.data)
            self.bias.data.div_(self.scale_bias.data)

    def scaling(self):
        weight = self.weight * self.scale_weight
        if hasattr(self, 'bias'):
            if self.scale_bias is not None:
                bias = self.bias * self.scale_bias
                return weight, bias
            else:
                return weight
        else:
            return weight
