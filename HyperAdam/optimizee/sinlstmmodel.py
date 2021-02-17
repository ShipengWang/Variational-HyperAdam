import torch
import torch.nn as nn
from optimizee.__init__ import Optimizee
import math
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

LOSS = nn.MSELoss()


class SinLSTMModel(Optimizee):
    def __init__(self, kwargs,
                 n_batches=128, n_h=20, n_l=10,
                 initial_param_scale=0.1,
                 ):
        for key_ in ['n_lstm', 'noise_scale']:
            if key_ not in kwargs:
                raise KeyError

        super().__init__()
        self.n_batches = n_batches
        self.n_h = n_h
        self.n_l = n_l
        self.n_lstm = kwargs['n_lstm']
        self.initial_param_scale = initial_param_scale
        self.noise_scale = kwargs['noise_scale']

        # self.add_module('lstm', nn.LSTM(input_size=1, hidden_size=self.n_h, num_layers=self.n_lstm))
        self.add_module('lstm0', MyLSTMCell(input_size=1, hidden_size=self.n_h))
        # self.hx = [torch.zeros(self.n_batches, self.n_h)]
        # self.cx = [torch.zeros(self.n_batches, self.n_h)]
        for i in range(self.n_lstm - 1):
            name = 'lstm' + str(i+1)
            self.add_module(name, MyLSTMCell(input_size=self.n_h, hidden_size=self.n_h))
            # self.hx.append([torch.zeros(self.n_batches, self.n_h)])
            # self.cx.append([torch.zeros(self.n_batches, self.n_h)])
        self.add_module('linear', nn.Linear(self.n_h, 1))

        self.reset_parameters()

        self.dim = self.get_dim()

    def reset_parameters(self):
        for module_ in self.modules():
            for para in module_.parameters():
                torch.nn.init.normal_(para, 0.0, self.initial_param_scale)

    def get_train_data_loader(self):
        pass

    def get_train_iter(self):
        pass

    def next_train_batch(self, device):
        x_data = np.zeros([self.n_l, self.n_batches, 1])
        y_data = np.zeros([self.n_batches, 1])
        for b in range(self.n_batches):
            phi = np.random.uniform(0.0, 2 * math.pi)
            omega = np.random.uniform(0.0, math.pi / 2)
            A = np.random.uniform(0.0, 10.0)
            for k in range(self.n_l):
                temp = A * math.sin(k * omega + phi) + np.random.normal(scale=self.noise_scale)
                x_data[k][b][0] = temp
            y_data[b][0] = A * math.sin(self.n_l * omega + phi)

        x_data = torch.from_numpy(x_data).float().to(device)
        y_data = torch.from_numpy(y_data).float().to(device)
        return x_data, y_data

    def forward(self, inputs=None):
        hx = [self.linear.bias.new_zeros(self.n_batches, self.n_h)]
        cx = [self.linear.bias.new_zeros(self.n_batches, self.n_h)]
        if self.n_lstm == 2:
            hx.append(self.linear.bias.new_zeros(self.n_batches, self.n_h))
            cx.append(self.linear.bias.new_zeros(self.n_batches, self.n_h))
        for l in range(self.n_l):
            hx[0], cx[0] = self.lstm0(inputs[l], [hx[0], cx[0]])
            if self.n_lstm == 2:
                hx[1], cx[1] = self.lstm1(hx[0], [hx[1], cx[1]])
        self.output = self.linear(hx[-1])

    def get_loss(self, data_batch=None):
        x_data, y_data = data_batch
        self.forward(x_data)
        loss = LOSS(self.output, y_data)
        return loss


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
        lstm_matrix = F.linear(torch.cat((inputs_, h_p_), dim=1), self.weight, self.bias)
        i, j, f, o = torch.chunk(lstm_matrix, 4, dim=1)
        # self._forget_bias acts the same with forget_bias in TensorFlow
        c = F.sigmoid(f + self._forget_bias) * c_p + F.sigmoid(i) * F.tanh(j)
        h = F.sigmoid(o) * F.tanh(c)
        return h, c


def main():
    device = torch.device('cuda')
    kwargs = {
        'n_lstm': 1,
        'noise_scale': 0.1
    }
    model = SinLSTMModel(kwargs)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.03125)
    losses = []
    for i in range(100):
        data_batch = model.next_train_batch(device)
        loss = model.get_loss(data_batch)
        model.get_grad(loss)
        optimizer.step()
        losses.append(loss.item())
    return losses


if __name__ == '__main__':
    main()
