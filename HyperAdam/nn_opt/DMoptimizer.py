from . import NNOptimizer

import torch
import torch.nn as nn


class DMoptimizer(NNOptimizer):
    def __init__(self, name,
                 num_layers=2, hidden_size=20, lr=0.1, weight_decay=0, pre_proc=True
                 ):
        super().__init__(name, weight_decay)
        self.learner = None
        self.learner_dim = None
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.add_module('lstm1', nn.LSTMCell(2, hidden_size))
        self.add_module('lstm2', nn.LSTMCell(hidden_size, hidden_size))
        for l in range(2, self.num_layers):
            name = 'lstm'+str(l+1)
            self.add_module(name, nn.LSTMCell(hidden_size, hidden_size))
        self.linear = nn.Linear(hidden_size, 1)
        self.hx = None
        self.cx = None

    @staticmethod
    def preproc(x, p=10.0):
        p = torch.tensor(p).to(x.device)
        xa = torch.Tensor.log(torch.Tensor.max(torch.Tensor.abs(x),
                              torch.Tensor.exp(-p))) / p
        xb = torch.Tensor.clamp(x * torch.Tensor.exp(p), min=-1.0, max=1.0)
        return torch.stack([xa, xb], dim=1)

    def forward(self, grad):
        # give the update vector with grad as input
        # preprocess
        x = self.preproc(grad)
        # lstm
        for i, module_ in enumerate(self.children()):
            if i == self.num_layers:
                continue
            self.hx[i], self.cx[i] = module_(x, (self.hx[i], self.cx[i]))
            x = self.hx[i]
        # fully connected
        out = self.linear(x)
        return out.squeeze()

    def reset_optimizer(self, keep_states=False, learner=None):
        if keep_states:
            for i in range(self.num_layers):
                self.hx[i] = self.hx[i].detach_()
                self.cx[i] = self.cx[i].detach_()
        else:
            self.learner_dim = learner.get_dim()
            self.hx = []
            self.cx = []
            for i in range(self.num_layers):
                self.hx.append(self.linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())
                self.cx.append(self.linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())

    # def reset_optimizer_learner(self, keep_states=False, leaner=None):
    #     self.reset_learner(leaner)
    #     self.reset_optimizer(keep_states=keep_states, leaner=leaner)

