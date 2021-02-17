import argparse
import torch
import warnings
import numpy as np
import pickle as p
from utils import *
import task_list
import os
import matplotlib.pyplot as plt
import json
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch learning-to-learn example')
parser.add_argument('--task', type=str, default='DMoptimizer',
                    help='choose optimizer to train')
parser.add_argument('--id', type=str, default='tetime',  # dbjifc
                    help='id')

parser.add_argument("--learner", type=str, default='mlp-sigmoid-mnist-9-100',
                    # mlp-sigmoid-mnist-10-100 cnn422-cifar-100
                    # mlp-sigmoid-mnist-8-10000 HyperAdam: lr 4e-4
                    help='set the learner to be trained')
parser.add_argument('--num', type=int, default=100, metavar='N',
                    help='the index of the training phase (default: 100)')

parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                    help='learning rate')

parser.add_argument('--eid', type=int, default=0, metavar='N',
                    help='the index of the training phase (default: 1)')

parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')

parser.add_argument('--device', type=str, default='cuda',
                    help='enables CUDA training')
parser.add_argument('--preproc', type=bool, default=True,
                    help='preprocess')
args = parser.parse_args()
device_ = torch.device(args.device)

# assert args.eid != 0
model_path, _, _, _, _ = set_model_path(task_id(args.task, args.id))


def log(s, filename):
    with open(filename, 'a') as fp:
        fp.write(s)


def train_with_learned_optimizer(optimizer, learner, steps, learning_rate):
    losses = []
    p = 0.0
    for s in range(steps):
        optimizer.reset_optimizer_learner(keep_states=s > 0, learner=learner)
        data_batch = learner.next_train_batch(device=device_)
        loss_prev = learner.get_loss(data_batch)
        # losses.append(loss_prev.data.cpu().numpy())
        learner.get_grad(loss_prev)
        s = time.time()
        with torch.no_grad():
            optimizer.step(learner, learning_rate=learning_rate)
        p = p + time.time() - s
        loss = learner.get_loss(data_batch)
        losses.append(loss.item())
    print(p / steps)
    return learner, losses


def train_with_learned_optimizer_time(optimizer, learner, steps, learning_rate):
    losses = []
    start = time.time()
    for s in range(steps):
        optimizer.reset_optimizer_learner(keep_states=s > 0, learner=learner)
        data_batch = learner.next_train_batch(device=device_)
        loss = learner.get_loss(data_batch)
        if loss < 0.6:
            print(time.time() - start)
            return learner, losses
        learner.get_grad(loss)
        with torch.no_grad():
            optimizer.step(learner, learning_rate=learning_rate)
        losses.append(loss.item())
    return learner, losses


def train_multiply_times(optimizer, learner, steps, learning_rate, task):
    losses = []
    for i in range(args.num):
        learner.get_train_data_loader()
        learner.get_train_iter()
        learner.reset_parameters()
        learner.to(device_)

        learner_in_meta_graph = get_learner(task['meta_test_set'][args.learner])
        optimizer.set_learner(learner_in_meta_graph, device=device_)

        learner_, loss = train_with_learned_optimizer(optimizer, learner, steps, learning_rate)
        losses.append(loss)
        print(args.learner, i, loss[-1])
    return losses


def main():
    tasks = task_list.tasks
    task = tasks[args.task]
    steps = task['meta_test_set'][args.learner]['steps']
    learning_rate = args.lr

    optimizer = task['optimizer'](name=task_id(args.task, args.id),
                                  num_layers=args.num_layers,
                                  hidden_size=args.hidden_size,
                                  # pre_proc=args.preproc
                                  )
    optimizer.eval()
    optimizer.to(device_)
    if args.eid > 0:
        optimizer.restore_model(args.eid, model_path)

    learner = get_learner(task['meta_test_set'][args.learner])
    losses = train_multiply_times(optimizer, learner, steps, learning_rate, task)
    avg_loss = np.mean(losses, axis=0)

    parent_path = os.path.split(os.getcwd())[0]
    data_path = os.path.join(parent_path, "test_data", args.learner, "all-" + args.task.split('-')[0])
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    name = args.task + "-" + args.id + "-num-" + str(args.num) + "-lr-" + \
           str(args.lr).replace('.', '') + "-step-" + str(steps) + ".pkl"
    file_name = os.path.join(data_path, name)

    print(avg_loss)
    plt.plot(avg_loss)
    plt.show(block=False)

    pickle_file = open(file_name, 'wb')
    p.dump(avg_loss, pickle_file)
    pickle_file.close()
    print(name + ": saved!")


if __name__ == '__main__':
    main()
