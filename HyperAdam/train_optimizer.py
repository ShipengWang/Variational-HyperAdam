import argparse
import sys
import numpy as np
import torch
import torch.optim as optim


from utils import *
import task_list

import warnings

import time
import matplotlib
matplotlib.rcParams["backend"] = "TKAgg"
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--task', type=str, default='HyperAdamVarSingleNorm',
                    help='choose optimizer to train')
parser.add_argument('--id', type=str, default=random_id(),  # random_id()
                    help='id')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 32)')

parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')

parser.add_argument('--update_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 700)')
parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                    help='number of epoch (default: 1)')
parser.add_argument('--eid', type=int, default=0, metavar='N',
                    help='the index of the training phase (default: 1)')

parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--preproc', type=bool, default=True,
                    help='preprocess')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='enables CUDA training')

parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()
print(vars(args))
device = torch.device(args.device)

assert args.optimizer_steps % args.truncated_bptt_step == 0
model_path, log_name, log_name_avg, exception_log, config_log = set_model_path(task_id(args.task, args.id))


def train_optimizer(task):
    # optimizer
    optimizer = task['optimizer'](name=task_id(args.task, args.id),
                                  num_layers=args.num_layers,
                                  hidden_size=args.hidden_size,
                                  pre_proc=args.preproc
                                  )
    optimizer.to(device)

    meta_optimizer = optim.Adam(optimizer.parameters(), lr=task['training_lr'], weight_decay=args.weight_decay)
    epoch = args.eid
    if epoch != 0:
        optimizer.restore_model(epoch, model_path)
    # figure
    gcf_dict = {}
    avg_final_losses = {}
    avg_all_losses = {}
    for i, key_ in enumerate(task['meta_train_set']):
        gcf_dict.update({key_: i+1})
        avg_final_losses.update({key_: []})
        avg_all_losses.update({key_: []})
    while args.max_epoch == 0 or epoch < args.max_epoch:
        start = time.time()
        epoch += 1
        meta_train_set_shuffled = shuffle_meta_train_set(task['meta_train_set'])
        for key_ in meta_train_set_shuffled:
            learner = get_learner(task['meta_train_set'][key_])
            learner_in_meta_graph = get_learner(task['meta_train_set'][key_])
            optimizer.set_learner(learner_in_meta_graph, device=device)
            learning_rate = task['meta_train_set'][key_]["lr"]
            #
            key_losses = []
            for i in range(args.update_per_epoch):
                start2 = time.time()
                # reinitialize the parameter of the learner
                learner.get_train_data_loader(shuffle=True)
                learner.reset_parameters()
                learner.to(device)
                # resample data from the dateset
                learner.get_train_iter()
                # loss of the learner
                losses = []
                avg_losses = []
                avg_log_losses = []
                for k in range(args.optimizer_steps // args.truncated_bptt_step):
                    # copy the parameters from learner to learner_in_meta_graph and
                    # set the state of LSTM if k==0 else detach the state if k>0
                    optimizer.reset_optimizer_learner(keep_states=k > 0, learner=learner)
                    loss_sum = 0.0
                    log_loss_sum = 0.0
                    for s in range(args.truncated_bptt_step):
                        data_batch = learner.next_train_batch(device)
                        loss_prev = learner.get_loss(data_batch)
                        learner.get_grad(loss_prev)
                        # loss_post = optimizer.learner.model.get_loss(data_batch)
                        learner_in_meta_graph_ = optimizer.step(learner, learning_rate=learning_rate)
                        loss_post = learner_in_meta_graph_.get_loss(data_batch)
                        # if loss_post > 10.0:
                        #     exception_msg = "\r loss is to large at epoch:{}, sample:{}".format(epoch, i)
                        #     log(exception_msg, exception_log)
                        #     exception_path = os.path.join(model_path, "exception")
                        #     if not os.path.exists(exception_path):
                        #         os.makedirs(exception_path)
                        #     optimizer.save_model(epoch, exception_path)
                        #     print("loss is large")
                        #     input("press any key to continue")
                        losses.append(loss_post.item())
                        loss_sum += loss_post
                        if hasattr(optimizer, 'kldloss'):
                            loss_sum += 0.0001 * optimizer.kldloss
                        # log_loss
                        # log_loss_sum += torch.log(loss_post + 1.0)
                    # update optimizer
                    optimizer.zero_grad()
                    avg_losses.append(loss_sum / args.truncated_bptt_step)
                    meta_loss = avg_losses[-1] if task["use_avg_loss"] else loss_post
                    # avg_log_losses.append(log_loss_sum / args.truncated_bptt_step)
                    # meta_loss = avg_log_losses[-1] if task["use_avg_loss"] else loss_post
                    meta_loss.backward()
                    meta_optimizer.step()

                end = time.time()
                du = end - start
                du2 = end - start2
                msg = "\repoch:{:2d}, task:{}, learner:{}, sample:{:2d}, " \
                      "avg_loss:{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, " \
                      "final_loss:{:.5f}; time:{:.5f}s".format(epoch, args.task + "-" + args.id,
                                                               key_, i, *avg_losses, losses[-1], du2)

                # print(msg)
                sys.stdout.write(msg)
                sys.stdout.flush()
                key_losses.append(losses)
                log(msg, log_name_avg)
                loss_str = [[str(losses[i]) for i in range(len(losses))]]
                msg = "\r{}".format(*loss_str)
                log(msg, log_name)
            # plot
            key_avg_loss = np.mean(key_losses, axis=0)
            avg_all_losses[key_].append(np.mean(key_avg_loss))
            avg_final_losses[key_].append(key_avg_loss[-1])
            key_msg = "\repoch:{:2d}, task:{}, learner:{}, avg_loss:{}".format(
                epoch, args.task + "-" + args.id, key_, key_avg_loss
            )
            print(key_msg)
            plt.figure(gcf_dict[key_])
            plt.ion()
            plt.plot(avg_all_losses[key_], "r")
            plt.plot(avg_final_losses[key_], "b")
            plt.legend(["avg_loss", "final_loss"])
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(args.task + "-" + args.id + "," + key_)
            # plt.draw()
            plt.show()
            plt.pause(0.001)
            plt.savefig(os.path.join(model_path, key_ + ".png"))
        if epoch % 10 == 0:
            optimizer.save_model(epoch, model_path)


def log(s, filename):
    with open(filename, 'a') as fp:
        fp.write(s)


def main():
    log_json(vars(args), config_log)
    tasks = task_list.tasks
    task = tasks[args.task]
    train_optimizer(task)


if __name__ == '__main__':
    main()
