import argparse
import warnings
import pickle as p
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.dataloader as loader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.optim

import massive_network
from utils import *
from nn_opt import *

import matplotlib

matplotlib.rcParams["backend"] = "TKAgg"
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch learning-to-learn example')
# Optimizer
parser.add_argument('--optimizer', type=str, default='adagrad', help='set the optimizer')
parser.add_argument('--opt_id', type=str, default='dbjifc', help='id, only for the learned optimizer')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--preproc', type=bool, default=True)
parser.add_argument('--load_opt', type=int, default=0, metavar='N', help='load optimizer')

# model
parser.add_argument('--learner', type=str, default='cnn1_mnist', help='set the learner to be trained')
parser.add_argument('--batch_size', type=float, default=1)

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--epoch', default=40, type=int)
parser.add_argument('--times', default=100, type=int)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--disp', default=20, type=int, help='print frequency')
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

args = parser.parse_args()

DEVICE = torch.device(args.device)
LEARNED_OPTER = {'HyperAdam': HyperAdam, 'HyperAdamVar': HyperAdamVar,
                 'rnnprop': RNNprop, 'DMoptimizer': DMoptimizer}


def main():
    opt_name = args.optimizer + '-' + args.opt_id if args.optimizer in LEARNED_OPTER.keys() else args.optimizer
    opt_name = opt_name + '-lr-' + str(args.lr).replace('.', '') + '-epoch-' + str(args.epoch) + '.pkl'
    print(args.optimizer)

    model = massive_network.__dict__[args.learner]()

    path_ = os.path.join('../test_cnn_longer', args.learner)
    print(path_)
    if not os.path.exists(path_):
        os.makedirs(path_)

    cudnn.benchmark = True

    losses = []
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = get_optimizer(model)
    for i in range(args.times):
        if 'cifar10' in args.learner:
            train_loader = get_data_loader('cifar10')
        elif 'mnist' in args.learner:
            train_loader = get_data_loader('mnist')
        else:
            raise ValueError('trainloader')

        model.reset_parameters()

        losses_per_time = []
        print("************{}-th***************".format(i))
        for epoch in range(args.epoch):
            loss = train(train_loader, model, criterion, optimizer, epoch)
            losses_per_time.append(loss)

        losses.append(losses_per_time)

    avg_loss = np.mean(losses, axis=0)
    print(avg_loss)
    plt.plot(avg_loss)
    plt.show(block=False)

    file_name = os.path.join(path_, opt_name)
    pickle_file = open(file_name, 'wb')
    p.dump(avg_loss, pickle_file)
    pickle_file.close()


def train(loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    for i, (inputs, target) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)

        output = model(inputs)
        loss = criterion(output, target)

        losses.update(loss.item(), inputs.size(0))

        model.zero_grad()
        loss.backward()
        if args.optimizer not in LEARNED_OPTER.keys():
            # optimizer.zero_grad()
            # loss.backward()
            optimizer.step()
        else:
            optimizer.reset_optimizer_learner(keep_states=(i > 0 or epoch > 0), learner=model)
            # loss.backward()
            with torch.no_grad():
                optimizer.step(model, learning_rate=args.lr)

        if i % args.disp == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), loss=losses
            ))
        return losses.avg


def get_data_loader(dataset):
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = loader.DataLoader(
            datasets.CIFAR10(root='../dataset/cifar10', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)
        return train_loader
    elif dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='../dataset/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, pin_memory=True)
        return train_loader


def get_optimizer(model):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # elif args.optimizer == 'adabound':
    #     optimizer = AdaBound(model.parameters(), args.lr, final_lr=args.final_lr, weight_decay=args.weight_decay)
    # elif args.optimizer == 'adashift':
    #     optimizer = AdaShift(model.parameters(), args.lr)
    elif args.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay, amsgrad=True)
    else:
        optimizer = LEARNED_OPTER[args.optimizer](name=task_id(args.optimizer, args.opt_id),
                                                  num_layers=args.num_layers, pre_proc=args.preproc,
                                                  hidden_size=args.hidden_size,
                                                  weight_decay=args.weight_decay)
        optimizer.eval()
        optimizer.to(DEVICE)
        if args.load_opt > 0:
            model_path, _, _, _, _ = set_model_path(task_id(args.optimizer, args.opt_id))
            optimizer.restore_model(args.load_opt, model_path)

        model_in_optimizer = massive_network.__dict__[args.arch]()
        optimizer.set_learner(model_in_optimizer, DEVICE)
    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
