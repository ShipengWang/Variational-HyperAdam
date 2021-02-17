import argparse
import shutil
import time
import warnings
import pickle as p

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.dataloader as loader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import massive_network
from utils import *
from nn_opt import *
from adam_opt import AdaBound

import matplotlib
matplotlib.rcParams["backend"] = "TKAgg"
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

model_names = sorted(name for name in massive_network.__dict__
                     if name.islower() and not name.startswith("__")
                     and (name.startswith("vgg") or name.startswith("resnet"))
                     and callable(massive_network.__dict__[name]))
# optimizer
parser = argparse.ArgumentParser(description='test hyperadam with resnet and vgg with imagenet and cifar')
parser.add_argument('--optimizer', default='HyperAdam', type=str,
                    help='set the optimizer')
parser.add_argument('--opt_id', type=str, default='massive-' + random_id(),
                    help='id')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--load_opt', type=int, default=100, metavar='N',
                    help='load optimizer')

parser.add_argument('--meta_epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_meta_epoch', default=0, type=int, metavar='N',
                    help='number of total epochs to run')
# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_cifar10',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet20_cifar10)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
# learning rate
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--adjust-every', dest='adjust_every', type=int, default=10,
                    help='adjust learning rate at every specified numper of epochs')
parser.add_argument('--adjust-rate', dest='adjust_rate', type=float, default=0.5)
# weight decay
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=5)
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()
best_acc1 = 0
device_ = torch.device(args.device)
learned_optimizer = {'HyperAdam': HyperAdam, 'rnnprop': RNNprop, 'DMoptimizer': DMoptimizer}
opt_name = args.optimizer + '-' + args.opt_id if args.optimizer in learned_optimizer.keys() else args.optimizer
path_ = os.path.join("../test_data_massive", args.arch, opt_name)


def train_optimizer():
    model = massive_network.__dict__[args.arch]()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer, model_path = get_initial_optimizer()
    meta_optimizer = torch.optim.Adam(optimizer.parameters(), lr=1e-4, weight_decay=0.0001)
    meta_epoch = args.start_meta_epoch
    init_model_every_epoch = 1
    losses = []
    while args.meta_epochs == 0 or meta_epoch < args.meta_epochs:
        meta_epoch += 1
        loss_ = []
        for i in range(init_model_every_epoch):
            model.reset_parameters()
            model.train()
            train_loader, val_loader = get_data_loader('cifar10')
            model.to(device_)
            for epoch in range(args.start_epoch, args.epochs):
                meta_loss = 0.0
                for k, (inputs, target) in enumerate(train_loader):
                    optimizer.reset_optimizer_learner(keep_states=(k > 0 or epoch > 0), learner=model)
                    inputs = inputs.to(device_)
                    target = target.to(device_)

                    # compute output
                    output = model(inputs)
                    loss_prev = criterion(output, target)

                    model.zero_grad()
                    loss_prev.backward()
                    learning_rate = adjust_learning_rate(optimizer, epoch)
                    model_in_opt_ = optimizer.step(model, learning_rate=learning_rate)

                    output_ = model_in_opt_(inputs)
                    loss_post = criterion(output_, target)
                    meta_loss += loss_post

                    loss_.append(float(loss_post))

                    if k > 0 and k % 20 == 0:
                        optimizer.zero_grad()
                        meta_loss = meta_loss / 20.0
                        meta_loss.backward()
                        meta_optimizer.step()
                        meta_loss = 0.0
        losses.append(loss_)

    if meta_epoch % 1 == 0:
        optimizer.save_model(meta_epoch, model_path)


def log(s, filename):
    with open(filename, 'a') as fp:
        fp.write(s)


def main():
    # check the save_dir exists or not
    global best_acc1, device_, learned_optimizer

    model = massive_network.__dict__[args.arch]()
    model.to(device_)

    if not os.path.exists(path_):
        os.makedirs(path_)
    file_name = os.path.join(path_, 'criterion' + '.pkl')
    model_path = os.path.join(path_, 'model')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at {}".format(args.resume))
        criterion_file = open(file_name, 'rb')
        criterion_dict = p.load(criterion_file)
    else:
        criterion_dict = {
            'training': {
                'loss': [],
                'top1': [],
                'top5': []
            },
            'validation': {
                'loss': [],
                'top1': [],
                'top5': []

            }
        }

    cudnn.benchmark = True
    if 'cifar10' in args.arch:
        train_loader, val_loader = get_data_loader('cifar10')
    elif 'cifar100' in args.arch:
        train_loader, val_loader = get_data_loader('cifar100')
    else:
        train_loader, val_loader = get_data_loader('ImageNet')

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()
    optimizer = get_optimizer(model)
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        loss_tr, top1_tr, top5_tr = train(train_loader, model, criterion, optimizer, epoch)
        criterion_dict['training']['loss'].append(loss_tr)
        criterion_dict['training']['top1'].append(top1_tr)
        criterion_dict['training']['top5'].append(top5_tr)
        # evaluate on validation set
        top1_val, top5_val, loss_val = validate(val_loader, model, criterion)
        criterion_dict['validation']['loss'].append(loss_val)
        criterion_dict['validation']['top1'].append(top1_val)
        criterion_dict['validation']['top5'].append(top5_val)

        # remember best acc@1 and save checkpoint
        is_best = top1_val > best_acc1
        best_acc1 = max(top1_val, best_acc1)

        if (epoch > 0 and epoch % args.save_every == 0) or epoch == args.epochs - 1:
            # save model
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1
            }, is_best, model_path, epoch)
            # save criterion
            pickle_file = open(file_name, 'wb')
            p.dump(criterion_dict, pickle_file)
            pickle_file.close()
        plt.figure(1)
        plt.ion()
        plt.plot(criterion_dict['training']['loss'], 'r')
        plt.plot(criterion_dict['validation']['loss'], 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['training', 'validation'])
        plt.title('loss:' + args.arch + ', optimizer:' + args.optimizer)
        plt.show()
        plt.pause(0.001)
        plt.savefig(os.path.join(model_path, 'loss-' + args.optimizer + '-' + args.opt_id + '.png'))


def get_data_loader(dataset):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader, val_loader = None, None
    if dataset == 'cifar10':
        train_loader = loader.DataLoader(
            datasets.CIFAR10(root='../dataset/cifar10', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = loader.DataLoader(
            datasets.CIFAR10(root='../dataset/cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    if dataset == 'cifar100':
        train_loader = loader.DataLoader(
            datasets.CIFAR10(root='../dataset/cifar100', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = loader.DataLoader(
            datasets.CIFAR10(root='../dataset/cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif dataset == 'ImageNet':
        train_loader = loader.DataLoader(
            datasets.ImageFolder(
                '../dataset/mini-ImageNet/train',
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = loader.DataLoader(
            datasets.ImageFolder('../dataset/mini-ImageNet/val', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_optimizer(model):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    else:
        optimizer = learned_optimizer[args.optimizer](name=task_id(args.optimizer, args.opt_id),
                                                      num_layers=args.num_layers,
                                                      hidden_size=args.hidden_size,
                                                      weight_decay=args.weight_decay)
        optimizer.to(device_)
        model_path, _, _, _ = set_model_path(task_id(args.optimizer, args.opt_id))
        optimizer.restore_model(args.load_opt, model_path)
    return optimizer


def get_initial_optimizer():
    optimizer = learned_optimizer[args.optimizer](name=task_id(args.optimizer, args.opt_id),
                                                  num_layers=args.num_layers,
                                                  hidden_size=args.hidden_size,
                                                  weight_decay=args.weight_decay)
    optimizer.to(device_)
    model_path, _, _, _ = set_model_path(task_id(args.optimizer, args.opt_id))
    model_in_optimizer = massive_network.__dict__[args.arch]()
    optimizer.set_learner(model_in_optimizer, device_)
    return optimizer, model_path


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device_)
        target = target.to(device_)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # compute gradient and do SGD step
        if args.optimizer not in learned_optimizer.keys():
            adjust_learning_rate(optimizer, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model_in_optimizer = massive_network.__dict__[args.arch]()
            optimizer.set_learner(model_in_optimizer, device_)
            optimizer.reset_optimizer_learner(keep_states=i > 0, learner=model)
            model.zero_grad()
            loss.backward()
            learning_rate = adjust_learning_rate(optimizer, epoch)
            optimizer.step(model, learning_rate=learning_rate)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         epoch, i, len(train_loader), batch_time=batch_time,
                         data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, float(top1.avg), float(top5.avg)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        target = target.to(device_)
        inputs = inputs.to(device_)

        if args.half:
            inputs = inputs.half()

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return float(top1.avg), float(top5.avg), losses.avg


def save_checkpoint(state, is_best, model_path, epoch):
    filename = os.path.join(model_path, 'checkpoint-' + str(epoch) + '.th')
    bestfilename = os.path.join(model_path, 'checkpoint-best.th')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestfilename)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.optimizer not in learned_optimizer.keys():
        lr = args.lr * (args.adjust_rate ** (epoch // args.adjust_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = args.lr * (args.adjust_rate ** (epoch // args.adjust_every))
        return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    train_optimizer()
