from nn_opt import *
from train_list import *
from test_list import meta_test_set


tasks = {
    'DMoptimizer': {
        'optimizer': DMoptimizer,
        'meta_train_set': meta_train_set,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
    'rnnprop': {
        'optimizer': RNNprop,
        'meta_train_set': meta_train_set,
        'meta_test_set': meta_test_set,
        'use_avg_loss': False,
        'training_lr': 1e-4,
    },
    'HyperAdam': {
        'optimizer': HyperAdam,
        'meta_train_set': meta_train_set,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
    'HyperAdamVar': {
        'optimizer': HyperAdamVar,
        'meta_train_set': meta_train_set,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
}
