import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from . import Optimizee  # from optimizee.__init__import
from optimizee.modules import ScaleConv2d, ScaleLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# no need to "softmax"
LOSS = nn.CrossEntropyLoss()


class ScaleCNN(Optimizee):
    def __init__(self, kwargs):
        # depth is the number of hidden layers
        super().__init__()
        for key_ in [
            'data_set', 'data_path', 'is_scaling',
            'fc_num', 'conv_num', 'pool_num', 'batch_size',
            'add_dropout', 'use_bn',
        ]:
            if key_ not in kwargs:
                raise KeyError

        self.scale = kwargs['is_scaling']
        self.data_set = kwargs['data_set']
        self.data_path = kwargs['data_path']
        self.fc_num = kwargs['fc_num']
        self.conv_num = kwargs['conv_num']
        self.pool_num = kwargs['pool_num']
        self.batch_size = kwargs['batch_size']
        self.add_dropout = kwargs['add_dropout']
        self.use_bn = kwargs['use_bn']

        self._get_activation('relu')

        assert self.conv_num % self.pool_num == 0
        assert self.data_set in self.data_path

        if self.data_set == 'cifar10':
            self.n_classes = 10
            self.input_size = 32
            self.input_channel = 3
        if self.data_set == 'mnist':
            self.n_classes = 10
            self.input_size = 28
            self.input_channel = 1

        for i in range(self.pool_num):
            name = 'conv_' + str(i) + str(0)
            in_channels = self.input_channel if i == 0 else 2 ** (i + 3)
            out_channels = 2 ** (i + 4)
            self.add_module(name, ScaleConv2d(in_channels, out_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              scale=self.scale))

            if self.use_bn:
                name = 'bn_' + str(i) + str(0)
                self.add_module(name, nn.BatchNorm2d(out_channels))
            for k in range(1, int(self.conv_num / self.pool_num)):
                name = 'conv_' + str(i) + str(k)
                in_channels = 2 ** (i + 4)
                out_channels = 2 ** (i + 4)
                self.add_module(name, ScaleConv2d(in_channels, out_channels,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  scale=self.scale))
                if self.use_bn:
                    name = 'bn_' + str(i) + str(k)
                    self.add_module(name, nn.BatchNorm2d(out_channels))

        self.fc_in_features = int(self.input_size * self.input_size * 2 ** 3 / 2 ** self.pool_num)
        for i in range(self.fc_num):
            name = 'fc_' + str(i)
            if i == 0:
                in_features = self.fc_in_features
            else:
                in_features = 2 ** (self.pool_num + 4)
            if i == self.fc_num - 1:
                out_features = self.n_classes
            else:
                out_features = 2 ** (self.pool_num + 4)
            self.add_module(name, ScaleLinear(in_features, out_features, scale=self.scale))

        self.dim = self.get_dim()

    def get_train_data_loader(self, cycling=True, shuffle=True):
        if self.data_set == 'mnist':
            self.train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(self.data_path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=self.batch_size, shuffle=shuffle, pin_memory=True)
        elif self.data_set == 'cifar10':
            self.train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(self.data_path, train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])),
                batch_size=self.batch_size, shuffle=shuffle, pin_memory=True)
        else:
            raise ValueError('dataset is mnist or cifar10')

        if cycling is True:
            self.cycling_data()

    def forward(self, inputs=None):
        features = inputs
        for j in range(self.pool_num):
            for k in range(int(self.conv_num / self.pool_num)):
                conv = self.__getattr__('conv_' + str(j) + str(k))
                features = conv(features)
                if self.use_bn:
                    bn = self.__getattr__('bn_' + str(j) + str(k))
                    features = bn(features)
                features = self.activationFunc(features)
            features = F.max_pool2d(features, 2, stride=2)

        last = features.view(-1, self.fc_in_features)
        for j in range(self.fc_num - 1):
            fc = self.__getattr__('fc_' + str(j))
            last = fc(last)
            last = self.activationFunc(last)
            if self.add_dropout:
                last = F.dropout(last, p=0.5)
        fc = self.__getattr__('fc_' + str(self.fc_num - 1))
        self.output = fc(last)

    def get_loss(self, data_batch=None):
        images, labels = data_batch
        self.forward(images)
        loss = LOSS(self.output, labels)
        return loss


def main():
    kwargs = {
        'data_set': 'cifar10',
        'data_path': '../../dataset/cifar10',
        'fc_num': 2,
        'pool_num': 2,
        'conv_num': 4,
        'batch_size': 128,
        'add_dropout': True,
        'use_bn': True,
        'is_scaling': True
    }
    model = ScaleCNN(kwargs)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.get_train_data_loader()
    for _ in range(3):
        model.reset_parameters()
        model.get_train_iter()
        for epoch in range(2):
            model.to(device)
            for i in range(100):
                model.next_train_batch(device)
                loss = model.get_loss()
                print(epoch, i, loss)
                model.get_grad(loss)
                optimizer.step()


if __name__ == '__main__':
    main()
