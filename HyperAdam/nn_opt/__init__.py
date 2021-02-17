from functools import reduce
from operator import mul
import torch
import torch.nn as nn
import os
import time


class NNOptimizer(nn.Module):
    def __init__(self, name, weight_decay=0):
        super().__init__()
        self.name = name
        self.learner = None
        self.lr = None
        self.regular_loss = None
        self.hx = None
        self.cx = None
        self.resume = True
        self.weight_decay = weight_decay
        self.learner_dim = None

    def set_learner(self, learner_in_meta_graph, device):
        # set the same learner in the meta graph
        # learner_in_meta_graph.reset_parameters()
        if self.training:
            learner_in_meta_graph.to(device)
            self.learner = LearnerInMetaGraph(learner_in_meta_graph)
        self.learner_dim = learner_in_meta_graph.get_dim()

    def get_dim(self):
        return self.learner_dim

    def forward(self, x):
        raise NotImplementedError

    def reset_optimizer(self, keep_states=False, learner=None):
        raise NotImplementedError

    def resume_optimizer(self, rnn_optimizer_state):
        self.hx = rnn_optimizer_state[0]
        self.cx = rnn_optimizer_state[1]

    def reset_learner(self, learner):
        self.learner.detach_var()
        self.learner.copy_params_from(learner)

    # def reset_optimizer_learner(self, keep_states=False, learner=None, rnn_optimizer_state=None):
    #     self.reset_learner(learner)
    #     if rnn_optimizer_state is not None and self.resume is True:
    #         self.resume_optimizer(rnn_optimizer_state)
    #         self.resume = False
    #     self.reset_optimizer(keep_states=keep_states, learner=learner)

    def reset_optimizer_learner(self, keep_states=False, learner=None, rnn_optimizer_state=None):
        if self.training:
            self.reset_learner(learner)
        if rnn_optimizer_state is not None and self.resume is True:
            self.resume_optimizer(rnn_optimizer_state)
            self.resume = False
        self.reset_optimizer(keep_states=keep_states, learner=learner)

    @staticmethod
    def get_flat_grads(model_with_grads):
        grads = []
        for param in model_with_grads.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads).squeeze(-1)

    # L2-regularization
    def get_flat_grads_L2_reg(self, model_with_grads):
        grads = []
        for param in model_with_grads.parameters():
            if param.grad is not None:
                grad_l2 = param.grad.data.add(self.weight_decay, param.data)
                grads.append(grad_l2.view(-1))
        return torch.cat(grads).squeeze(-1)

    def step(self, model_with_grads, learning_rate=None):
        lr = self.lr if learning_rate is None else learning_rate

        if self.training:
            flat_params = self.learner.get_flat_params(model_with_grads)
        else:
            flat_params = self.get_flat_params(model_with_grads)

        # L2 regularization
        flat_grads = self.get_flat_grads_L2_reg(model_with_grads)
        # s = time.time()
        update_ = self.forward(flat_grads)
        # print("intern time:{}".format(time.time()-s))
        flat_params = flat_params - lr * update_

        if self.training:
            self.learner.set_flat_params(flat_params)
            self.learner.copy_params_to(model_with_grads)
            return self.learner.model
        else:
            self.register_param_to_model(model_with_grads, flat_params)


    @staticmethod
    def get_flat_params(model):
        params = []
        for param in model.parameters():
            if param.requires_grad is True:
                params.append(param.view(-1))
        return torch.cat(params).squeeze(-1)

    @staticmethod
    def register_param_to_model(model, flat_params):
        offset = 0
        for param in model.parameters():
            param_shape = param.size()
            param_flat_size = reduce(mul, param_shape, 1)
            param.data = flat_params[offset: offset + param_flat_size].view(*param_shape)
        return model

    def save_model(self, epoch, model_path):
        filename = "params-epoch-%d.pkl" % epoch
        torch.save(self.state_dict(), os.path.join(model_path, filename))

    def restore_model(self, checkpoint, model_path):
        filename = "params-epoch-%d.pkl" % checkpoint
        self.load_state_dict(torch.load(os.path.join(model_path, filename), map_location='cuda:0'))


class LearnerInMetaGraph:
    # put learner in the meta model graph
    def __init__(self, model):
        self.model = model

    def detach_var(self):
        for module_ in self.model.modules():
            if hasattr(module_, 'weight') and module_.weight is not None:
                module_.weight = nn.Parameter(module_.weight).detach_()
                if hasattr(module_, 'bias'):
                    if module_.bias is not None:
                        module_.bias = nn.Parameter(module_.bias).detach_()

    def get_flat_params(self, model):
        params = []
        for paramA, paramB in zip(self.model.parameters(), model.parameters()):
            if paramB.requires_grad is True:
                params.append(paramA.view(-1))
        return torch.cat(params).squeeze(-1)

    def set_flat_params(self, flat_params):
        offset = 0
        for module_ in self.model.modules():
            if hasattr(module_, 'weight') and module_.weight is not None:
                weight_shape = module_.weight.size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                # should not be leaf node
                module_._parameters['weight'] = flat_params[
                                                offset:offset + weight_flat_size].view(*weight_shape)
                offset += weight_flat_size

                if module_.bias is not None:
                    bias_shape = module_.bias.size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module_._parameters['bias'] = flat_params[
                                                  offset:offset + bias_flat_size].view(
                        *bias_shape)
                    offset += bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            if modelB.requires_grad is True:
                modelB.data.copy_(modelA.data)


from .DMoptimizer import DMoptimizer
from .rnnprop import RNNprop
from .HyperAdam import HyperAdam
from .HyperAdamConf import HyperAdamConf
from .HyperAdamVar import HyperAdamVar
from .HyperAdamVarSingle import HyperAdamVarSingle
from .HyperAdamVarSingleNorm import HyperAdamVarSingleNorm

__all__ = ['DMoptimizer', 'RNNprop', 'HyperAdam', 'HyperAdamVar']
