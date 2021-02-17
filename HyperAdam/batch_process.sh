#!/usr/bin/env bash

############## train optimizer ##########################
python ./train_optimizer.py --task HyperAdamVar --hidden_size 20


############## test optimizer ##########################
python ./test_optimizer_toy.py --task=HyperAdamVar --id=dbjifc --learner=cnn211-cifar-2000 --lr=3e-5

EPOCH=200
LEARNER=cnn1_cifar10
python ./test_optimizer_longer.py --epoch $EPOCH --optimizer HyperAdamVar --opt_id dbjifc --load_opt 100  --learner $LEARNER --lr 1e-4 --weight_decay 1e-4 --adjust_rate 0.5 --adjust_every 100

python ./test_optimizer_massive.py --optimizer HyperAdamVar --opt_id dbjifc  --arch resnet32_cifar10 --lr 1e-5 --weight_decay 1e-4
