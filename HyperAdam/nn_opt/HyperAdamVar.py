import torch
import torch.nn.functional as F
from . import NNOptimizer

from .cells import MyLinear, MyLSTMCell, AdamCell

muf = []
stdf = []


class HyperAdamVar(NNOptimizer):
    def __init__(self, name,
                 num_layers=2, hidden_size=20, lr=0.005, weight_decay=0, pre_proc=True,
                 use_var_cell = True
                 ):
        super().__init__(name, weight_decay)
        self.learner = None
        self.learner_dim = None
        self.lr = lr
        self.hidden_size = hidden_size
        self.lstm_layers = num_layers - 1
        self.pre_proc = pre_proc
        self.use_var_cell = use_var_cell

        self.b1 = 0.95
        self.b2 = 0.95
        self.eps = 1e-8
        self.m = None
        self.v = None
        self.b1t = None
        self.b2t = None

        if self.pre_proc:
            self.pre_linear = MyLinear(2, hidden_size)  # 1
            input_size = hidden_size
        else:
            input_size = 2

        for i in range(self.lstm_layers):
            name = 'lstm' + str(i + 1)
            self.add_module(name, MyLSTMCell(input_size, hidden_size))
            input_size = hidden_size

        self.add_module("adamcell", AdamCell(input_size, hidden_size))
        self.post_linear = MyLinear(input_size, hidden_size)
        if self.use_var_cell:
            self.linear_var1 = MyLinear(input_size, 1)

        self.hx = None
        self.cx = None
        self.kldloss = 0

        self.reset_parameters()

    def reset_parameters(self):
        for module_ in self.children():
            module_.reset_parameters()

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, grad):
        # norm_grad = grad_normalization(grad)
        # output the update vector with grad as input

        self.m = self.b1 * self.m + (1 - self.b1) * grad if self.m is not None else (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2) if self.v is not None else (1 - self.b2) * (grad ** 2)
        self.b1t = self.b1 if self.b1t is None else self.b1t * self.b1
        self.b2t = self.b2 if self.b2t is None else self.b2t * self.b2
        sv = torch.sqrt(self.v / (1 - self.b2t))
        sm = self.m / (1 - self.b1t)

        # preproc, modules[0]
        if self.pre_proc:
            last_ = self.pre_linear(torch.stack([grad / (sv + self.eps), sm / (sv + self.eps)], 1))
            last = F.elu(last_)
        else:
            last = torch.stack([grad / (sv + self.eps), sm / (sv + self.eps)], 1)

        for i, module_ in enumerate(self.filter_children("lstm")):
            self.hx[i], self.cx[i] = module_(last, (self.hx[i], self.cx[i]))
            last = self.hx[i]

        # AdamCell
        adam, self.hx[-1], self.cx[-1] = self.adamcell([torch.unsqueeze(grad, 1), last],
                                                       (self.hx[-1], self.cx[-1]))
        # WeightCell
        if self.hidden_size > 1:
            weight_ = F.elu(self.post_linear(last))
            mu = torch.sum(adam * weight_, dim=1)
        else:
            mu = torch.squeeze(adam)
        if self.use_var_cell:
            # varcell
            update_var = self.linear_var1(last)

            log_var = update_var.view(-1)
            update_f = self.reparameterize(mu, log_var)
        else:
            update_f = mu

        # var1 = torch.ones_like(sv)  ## norm_grad ** 2(yangyan)
        # var0 = log_var.exp()
        # kld = 1 + log_var - torch.log((var1 + 1e-6)) - var0.div(var1 + 1e-6)
        # self.kldloss = - 0.5 * torch.sum(kld)
        if self.training:
            # prior: N(sm, 1)
            var0 = torch.ones_like(sv)
            var1 = log_var.exp()
            # kldloss = torch.log(torch.sqrt(var0) + 1e-6) - torch.log(torch.sqrt(var1 + 1e-12)) \
            #                + (var1 + (mu - sm / (sv + self.eps)) ** 2) / (2 * var0 + 1e-6) - 0.5   # sm / (sv + self.eps)
            #
            # (var1 + (mu - sm / (sv + self.eps)) ** 2)
            kldloss = - 0.5 * log_var + var1 / (2 * var0 + 1e-6) - 0.5 + torch.log(torch.sqrt(var0) + 1e-6)
            # independent among different coordinates
            # mean, instead of sum, can be treated as an adaptive hyperparameter
            self.kldloss = torch.sum(kldloss)
        return update_f

    def filter_children(self, filter):
        for name, module in self.named_children():
            if filter in name:
                yield module

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
                self.hx.append(self.post_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())
                self.cx.append(self.post_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_())
            self.hx.append([self.post_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_(),
                            self.post_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_()])
            self.cx.append([self.post_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_(),
                            self.post_linear.bias.new_zeros(self.learner_dim, self.hidden_size).detach_()])
