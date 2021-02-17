from .VGGNet import VGG
from .ResNet import ResNet, ResNetCifar
from .SimpleCNN import SimpleCNN

import torch

cfg_vgg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

__all__ = [
              'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
          ] \
          + [
              'vgg11_cifar10', 'vgg11_bn_cifar10', 'vgg13_cifar10', 'vgg13_bn_cifar10', 'vgg16_cifar10',
              'vgg16_bn_cifar10', 'vgg19_bn_cifar10', 'vgg19_cifar10',
          ] \
          + [
              'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
          ] \
          + [
              'resnetcifar20', 'resnetcifar32', 'resnetcifar44', 'resnetcifar56', 'resnetcifar110'
          ] \
          + [
              'cnn1_cifar10', 'cnn1_mnist', 'cnn2_cifar10', 'cnn2_mnist'
          ] \
          + [
              'cnn1_cifar10_bn', 'cnn1_mnist_bn', 'cnn2_cifar10_bn', 'cnn2_mnist_bn'
          ] \
          + ['cnn2_cifar10_dropout', 'cnn2_mnist_dropout']


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg_vgg['A'], 'ImageNet')


def vgg11_cifar10():
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg_vgg['A'], 'Cifar10')


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg_vgg['A'], 'ImageNet', batch_norm=True)


def vgg11_bn_cifar10():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg_vgg['A'], 'Cifar10', batch_norm=True)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(cfg_vgg['B'], 'ImageNet')


def vgg13_cifar10():
    """VGG 13-layer model (configuration "B")"""
    return VGG(cfg_vgg['B'], 'Cifar10')


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(cfg_vgg['B'], 'ImageNet', batch_norm=True)


def vgg13_bn_cifar10():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(cfg_vgg['B'], 'Cifar10', batch_norm=True)


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(cfg_vgg['D'], 'ImageNet')


def vgg16_cifar10():
    """VGG 16-layer model (configuration "D")"""
    return VGG(cfg_vgg['D'], 'Cifar10')


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg_vgg['D'], 'ImageNet', batch_norm=True)


def vgg16_bn_cifar10():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg_vgg['D'], 'Cifar10', batch_norm=True)


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(cfg_vgg['E'], 'ImageNet')


def vgg19_cifar10():
    """VGG 19-layer model (configuration "E")"""
    return VGG(cfg_vgg['E'], 'Cifar10')


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(cfg_vgg['E'], 'ImageNet', batch_norm=True)


def vgg19_bn_cifar10():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(cfg_vgg['E'], 'Cifar10', batch_norm=True)


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet('BasicBlock', [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet('BasicBlock', [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet('BasicBlock', [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet('Bottleneck', [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet('Bottleneck', [3, 8, 36, 3], **kwargs)
    return model


def resnet20_cifar10():
    return ResNetCifar('BasicBlock', [3, 3, 3])


def resnet32_cifar10():
    return ResNetCifar('BasicBlock', [5, 5, 5])


def resnet44_cifar10():
    return ResNetCifar('BasicBlock', [7, 7, 7])


def resnet56_cifar10():
    return ResNetCifar('BasicBlock', [9, 9, 9])


def resnet110_cifar10():
    return ResNetCifar('BasicBlock', [18, 18, 18])


####################################################

def cnn1_cifar10():
    kwargs = {
        'data_set': 'cifar10',
        'conv_num': 2,
        'fc_num': 1,
        'pool_num': 1,
        'add_dropout': False,
        'use_bn': False,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn1_mnist():
    kwargs = {
        'data_set': 'mnist',
        'conv_num': 2,
        'fc_num': 1,
        'pool_num': 1,
        'add_dropout': False,
        'use_bn': False,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn2_cifar10():
    kwargs = {
        'data_set': 'cifar10',
        'conv_num': 4,
        'fc_num': 2,
        'pool_num': 2,
        'add_dropout': False,
        'use_bn': False,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn2_mnist():
    kwargs = {
        'data_set': 'mnist',
        'conv_num': 4,
        'fc_num': 2,
        'pool_num': 2,
        'add_dropout': False,
        'use_bn': False,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn1_cifar10_bn():
    kwargs = {
        'data_set': 'cifar10',
        'conv_num': 2,
        'fc_num': 1,
        'pool_num': 1,
        'add_dropout': False,
        'use_bn': True,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn1_mnist_bn():
    kwargs = {
        'data_set': 'mnist',
        'conv_num': 2,
        'fc_num': 1,
        'pool_num': 1,
        'add_dropout': False,
        'use_bn': True,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn2_cifar10_bn():
    kwargs = {
        'data_set': 'cifar10',
        'conv_num': 4,
        'fc_num': 2,
        'pool_num': 2,
        'add_dropout': False,
        'use_bn': True,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn2_mnist_bn():
    kwargs = {
        'data_set': 'mnist',
        'conv_num': 4,
        'fc_num': 2,
        'pool_num': 2,
        'add_dropout': False,
        'use_bn': True,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn2_cifar10_dropout():
    kwargs = {
        'data_set': 'cifar10',
        'conv_num': 4,
        'fc_num': 2,
        'pool_num': 2,
        'add_dropout': True,
        'use_bn': False,
    }
    return SimpleCNN(kwargs=kwargs)


def cnn2_mnist_dropout():
    kwargs = {
        'data_set': 'mnist',
        'conv_num': 4,
        'fc_num': 2,
        'pool_num': 2,
        'add_dropout': True,
        'use_bn': False,
    }
    return SimpleCNN(kwargs=kwargs)
