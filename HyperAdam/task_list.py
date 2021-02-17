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
    'HyperAdamVarSingle': {
        'optimizer': HyperAdamVarSingle,
        'meta_train_set': meta_train_set,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },

    'HyperAdamVarSingleNorm': {
        'optimizer': HyperAdamVarSingleNorm,
        'meta_train_set': meta_train_set_scale,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
    ########Ablation Study########
    'HyperAdamVarNeither': {
        'optimizer': HyperAdamVar,
        'meta_train_set': meta_train_set_neither,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
    'HyperAdamVarConvex': {
        'optimizer': HyperAdamVar,
        'meta_train_set': meta_train_set_convex,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
    'HyperAdamVarScale': {
        'optimizer': HyperAdamVar,
        'meta_train_set': meta_train_set_scale,
        'meta_test_set': meta_test_set,
        'use_avg_loss': True,
        'training_lr': 1e-4,
    },
}
