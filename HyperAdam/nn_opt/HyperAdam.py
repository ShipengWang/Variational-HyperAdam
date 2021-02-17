import torch
import torch.nn.functional as F

from . import NNOptimizer
from .cells import MyLinear, MyLSTMCell, AdamCell, grad_normalization


class HyperAdam(NNOptimizer):
    def __init__(self, name,
                 num_layers=2, hidden_size=20, lr=0.005, weight_decay=0, pre_proc=True
                 ):
        super().__init__(name, weight_decay)
        self.learner = None
        self.learner_dim = None
        self.lr = lr
        self.hidden_size = hidden_size
        self.lstm_layers = num_layers - 1
        self.pre_linear = MyLinear(1, hidden_size)  # 1
        for i in range(self.lstm_layers):
            name = 'lstm' + str(i + 1)
            self.add_module(name, MyLSTMCell(hidden_size, hidden_size))
        self.add_module("adamcell", AdamCell(hidden_size, hidden_size))
        self.post_linear = MyLinear(hidden_size, hidden_size)
        self.hx = None
        self.cx = None
        self.reset_parameters()

    def reset_parameters(self):
        for module_ in self.children():
            module_.reset_parameters()

    def forward(self, grad):
        norm_grad = grad_normalization(grad)
        modules = list(self.children())
        # preproc, modules[0]
        last_ = self.pre_linear(torch.stack([norm_grad], 1))
        # deepmind
        # preproc = self.preproc(norm_grad)
        # last_ = self.pre_linear(preproc)

        last = F.elu(last_)
        # LSTM modules[1:-2]
        for i, module_ in enumerate(modules[1:-2]):
            self.hx[i], self.cx[i] = module_(last, (self.hx[i], self.cx[i]))
            last = self.hx[i]
        # AdamCell
        adam, self.hx[-1], self.cx[-1] = self.adamcell([torch.unsqueeze(grad, 1), last],
                                                                    (self.hx[-1], self.cx[-1]))
        # WeightCell
        weight_ = F.elu(self.post_linear(last))
        update = torch.sum(adam * weight_, dim=1)
        # w = weight_ / torch.sum(weight_, dim=1, keepdim=True)
        # update = torch.sum(adam * w, dim=1)
        return update

    def reset_optimizer(self, keep_states=False, learner=None):
        if keep_states:
            for i in range(self.lstm_layers):
                self.hx[i] = self.hx[i].detach_()
                self.cx[i] = self.cx[i].detach_()
            self.hx[-1] = [self.hx[-1][0].detach_(), self.hx[-1][1].detach_()]
            self.cx[-1] = [self.cx[-1][0].detach_(), self.cx[-1][1].detach_()]
        else:
            self.learner_dim = learner.get_dim()
            self.hx = []
            self.cx = []
            # new_zeros: creating a tensor with size filled with 0, the same dtype and device
            for i in range(self.lstm_layers):
                self.hx.append(self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())
                self.cx.append(self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())
            self.hx.append([self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_(),
                            self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_()])
            self.cx.append([self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_(),
                            self.pre_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_()])
