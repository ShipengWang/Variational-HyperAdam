import optimizee

meta_test_set = {
    'mlp-sigmoid-mnist-1-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 1,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-2-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 2,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-3-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 3,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-4-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 4,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-5-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 5,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-6-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 6,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-7-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 7,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-8-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 8,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-9-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                'steps': 100,
                                'kwargs': {
                                    'depth': 9,
                                    'activation': 'sigmoid',
                                    'is_scaling': None,
                                    'data_path': '../dataset/mnist',
                                    'batch_size': 128
                                },  # number of hidden layers, activation, is_scaling
                                },
    'mlp-sigmoid-mnist-10-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                 'steps': 100,
                                 'kwargs': {
                                     'depth': 10,
                                     'activation': 'sigmoid',
                                     'is_scaling': None,
                                     'data_path': '../dataset/mnist',
                                     'batch_size': 128
                                 },  # number of hidden layers, activation, is_scaling
                                 },
    'mlp-sigmoid-mnist-9-10000': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                  'steps': 10000,
                                  'kwargs': {
                                      'depth': 9,
                                      'activation': 'sigmoid',
                                      'is_scaling': None,
                                      'data_path': '../dataset/mnist',
                                      'batch_size': 128
                                  },  # number of hidden layers, activation, is_scaling
                                  },
    'mlp-sigmoid-mnist-8-10000': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                                      'steps': 10000,
                                      'kwargs': {
                                          'depth': 8,
                                          'activation': 'sigmoid',
                                          'is_scaling': None,
                                          'data_path': '../dataset/mnist',
                                          'batch_size': 128
                                      },  # number of hidden layers, activation, is_scaling
                                      },
    ###############################################################################
    'mlp-relu-mnist-1-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                             'steps': 100,
                             'kwargs': {
                                 'depth': 1,
                                 'activation': 'relu',
                                 'is_scaling': None,
                                 'data_path': '../dataset/mnist',
                                 'batch_size': 128
                             },  # number of hidden layers, activation, is_scaling
                             },
    'mlp-elu-mnist-1-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                            'steps': 100,
                            'kwargs': {
                                'depth': 1,
                                'activation': 'elu',
                                'is_scaling': None,
                                'data_path': '../dataset/mnist',
                                'batch_size': 128
                            },  # number of hidden layers, activation, is_scaling
                            },
    'mlp-tanh-mnist-1-100': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                             'steps': 100,
                             'kwargs': {
                                 'depth': 1,
                                 'activation': 'tanh',
                                 'is_scaling': None,
                                 'data_path': '../dataset/mnist',
                                 'batch_size': 128
                             },  # number of hidden layers, activation, is_scaling
                             },
    #################################################################################
    'cnn211-mnist-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn211-cifar-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn211-cifar-bn-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn211-mnist-bn-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    # No need to add dropout to cnn211 which only has 1 fully-connected layer
    'cnn422-mnist-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-cifar-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-cifar-bn-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-mnist-bn-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-cifar-dropout-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': True,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-mnist-dropout-100': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 100,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': True,
            'use_bn': False,
            'is_scaling': None
        }
    },
    ################################################################
    'cnn211-mnist-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn211-cifar-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn211-cifar-bn-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn211-mnist-bn-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    ################################################################
    'cnn211-mnist-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn211-cifar-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn211-cifar-bn-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn211-mnist-bn-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 2,
            'fc_num': 1,
            'pool_num': 1,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-mnist-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-cifar-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-cifar-bn-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-mnist-bn-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-cifar-dropout-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': True,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-mnist-dropout-10000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 10000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': True,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-mnist-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-cifar-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-cifar-bn-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-mnist-bn-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': False,
            'use_bn': True,
            'is_scaling': None
        }
    },
    'cnn422-cifar-dropout-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'cifar10',
            'data_path': '../dataset/cifar10',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': True,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'cnn422-mnist-dropout-2000': {
        'learner': optimizee.cnn.ScaleCNN,
        'steps': 2000,
        'kwargs': {
            'data_set': 'mnist',
            'data_path': '../dataset/mnist',
            'conv_num': 4,
            'fc_num': 2,
            'pool_num': 2,
            'batch_size': 128,
            'add_dropout': True,
            'use_bn': False,
            'is_scaling': None
        }
    },
    'lstm-base-100': {
        'learner': optimizee.sinlstmmodel.SinLSTMModel,
        'steps': 100,
        'kwargs': {
            'n_lstm': 1,
            'noise_scale': 0.1
        }
    },
    'lstm-x2-100': {
        'learner': optimizee.sinlstmmodel.SinLSTMModel,
        'steps': 100,
        'kwargs': {
            'n_lstm': 2,
            'noise_scale': 0.1
        }
    },

    'lstm-x2-10000': {
        'learner': optimizee.sinlstmmodel.SinLSTMModel,
        'steps': 10000,
        'kwargs': {
            'n_lstm': 2,
            'noise_scale': 0.1
        }
    },
    'lstm-smallnoise-100': {
        'learner': optimizee.sinlstmmodel.SinLSTMModel,
        'steps': 100,
        'kwargs': {
            'n_lstm': 1,
            'noise_scale': 0.01
        }
    },
}
