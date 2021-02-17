from . import NNOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time


class RNNprop(NNOptimizer):
    def __init__(self, name,
                 num_layers=2, hidden_size=20, lr=0.1, weight_decay=0, pre_proc=True
                 ):
        super().__init__(name, weight_decay)
        self.learner = None
        self.learner_dim = None
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.b1 = 0.95
        self.b2 = 0.95
        self.eps = 1e-8
        self.m = None
        self.v = None
        self.b1t = None
        self.b2t = None

        self.pre_linear = nn.Linear(2, hidden_size)
        self.add_module('lstm1', nn.LSTMCell(hidden_size, hidden_size))
        self.add_module('lstm2', nn.LSTMCell(hidden_size, hidden_size))
        for l in range(2, self.num_layers):
            name = 'lstm'+str(l+1)
            self.add_module(name, nn.LSTMCell(hidden_size, hidden_size))
        self.post_linear = nn.Linear(hidden_size, 1)
        self.hx = None
        self.cx = None

    def forward(self, grad):
        # output the update vector with grad as input
        # s = time.time()
        self.m = self.b1 * self.m + (1 - self.b1) * grad if self.m is not None else (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2) if self.v is not None else (1 - self.b2) * (grad ** 2)
        self.b1t = self.b1 if self.b1t is None else self.b1t * self.b1
        self.b2t = self.b2 if self.b2t is None else self.b2t * self.b2
        sv = torch.sqrt(self.v / (1 - self.b2t)) + self.eps
        x_ = torch.stack([grad / sv, (self.m / (1 - self.b1t)) / sv], 1)
        # print("intern time: {}".format(time.time()-s))
        # preproc
        x = F.elu(self.pre_linear(x_))
        # lstm
        i = 0
        for name, module_ in self.named_children():
            if "lstm" in name:
                self.hx[i], self.cx[i] = module_(x, (self.hx[i], self.cx[i]))
                x = self.hx[i]
                i = i + 1
        # fully connected
        out = self.post_linear(x)
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

            # self.m = torch.zeros([self.learner_dim])
            # self.v = torch.zeros([self.learner_dim])
            # self.b1t = torch.ones([self.learner_dim])
            # self.b2t = torch.ones([self.learner_dim])

            for i in range(self.num_layers):
                self.hx.append(self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())
                self.cx.append(self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())

    # def reset_optimizer_learner(self, keep_states=False, leaner=None):
    #     self.reset_learner(leaner)
    #     self.reset_optimizer(keep_states=keep_states, leaner=leaner)


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, forget_bias=1.0):
        """

        :param input_size:
        :param hidden_size:
        :param bias:
        :param forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints. (from TensorFlow)
        """
        # forget_bias acts the same with LSTMCell in TensorFlow v1.5
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_bias = bias
        self._forget_bias = forget_bias
        self.weight = Parameter(torch.Tensor(4 * hidden_size, input_size + hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
            # self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, inputs, hx=None):
        c_p, m_p = hx
        lstm_matrix = F.linear(torch.cat((inputs, m_p), dim=1), self.weight, self.bias)
        i, j, f, o = torch.chunk(lstm_matrix, 4, dim=1)
        # self._forget_bias acts the same with forget_bias in TensorFlow
        c = F.sigmoid(f + self._forget_bias) * c_p + F.sigmoid(i) * F.tanh(j)
        m = F.sigmoid(o) * F.tanh(c)
        return m, c
