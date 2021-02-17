from . import NNOptimizer
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def grad_normalization(x, epsilon=1e-16):
    norm = torch.norm(x, p=2, dim=0)
    inv = torch.rsqrt(norm ** 2 + epsilon)
    y = x * inv
    return y


def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2.0) & (tmp > -2.0)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class MyLinear(nn.Linear):
    def reset_parameters(self):
        # nn.init.xavier_normal_(self.weight)
        truncated_normal_(self.weight.data, std=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


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

    def forward(self, inputs_, hcx=None):
        h_p_, c_p = hcx
        #
        inputs = grad_normalization(inputs_) # inputs_  #  grad_normalization(inputs_)
        h_p = grad_normalization(h_p_)  # h_p_   # grad_normalization(h_p_)

        lstm_matrix = F.linear(torch.cat((inputs, h_p), dim=1), self.weight, self.bias)
        i, j, f, o = torch.chunk(lstm_matrix, 4, dim=1)
        # self._forget_bias acts the same with forget_bias in TensorFlow
        c = F.sigmoid(f + self._forget_bias) * c_p + F.sigmoid(i) * F.tanh(j)
        h = F.sigmoid(o) * F.tanh(c)
        return h, c


class AdamCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, eps=1e-24):
        super(AdamCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_bias = bias
        self.temperature = 1.0  # / math.sqrt(hidden_size)
        self.eps = eps
        # self.weight_ih = Parameter(torch.Tensor(2 * hidden_size, input_size))
        # self.weight_hh = Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.weight = Parameter(torch.Tensor(2 * hidden_size, input_size + hidden_size))
        if bias:
            self.bias = Parameter(torch.Tensor(2 * hidden_size))
            # self.bias_ih = Parameter(torch.Tensor(2 * hidden_size))
            # self.bias_hh = Parameter(torch.Tensor(2 * hidden_size))
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
        # input = [gradident, output of lstm]
        grad, state = inputs
        m_t, v_t = hx[0]
        beta_t, gamma_t = hx[1]

        # self.check_forward_input(state)
        # self.check_forward_hidden(state, mt, '[0]')
        # self.check_forward_hidden(state, betat, '[1]')

        # normalization
        norm_m = grad_normalization(m_t)
        norm_state = grad_normalization(state)  # may be not

        # first-moment and second-moment
        # exponential decay rates
        # decay_rates_ = F.linear(norm_state, self.weight_ih, self.bias_ih) + F.linear(norm_m,
        #                                                                              self.weight_hh, self.bias_hh)
        # decay_rates_ = F.linear(torch.cat((norm_m, state), dim=1), self.weight, self.bias)

        decay_rates_ = F.linear(torch.cat((norm_m, norm_state), dim=1), self.weight, self.bias)

        decay_rates = F.sigmoid(decay_rates_ / self.temperature)
        decay_rates_m, decay_rates_v = torch.chunk(decay_rates, 2, dim=1)

        m_t_ = decay_rates_m * m_t + (1 - decay_rates_m) * grad
        v_t_ = decay_rates_v * v_t + (1 - decay_rates_v) * (grad ** 2)
        beta_t_ = decay_rates_m * beta_t + (1 - decay_rates_m)
        gamma_t_ = decay_rates_v * gamma_t + (1 - decay_rates_v)

        m_t_hat = m_t_ / (beta_t_ + self.eps)
        v_t_hat = v_t_ / (gamma_t_ + self.eps)

        ada_m_t = m_t_hat / torch.sqrt(v_t_hat + self.eps)
        return ada_m_t, [m_t_, v_t_], [beta_t_, gamma_t_]
