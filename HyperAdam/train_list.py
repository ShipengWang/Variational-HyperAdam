import optimizee

# both tricks
meta_train_set = {
    'MLP_tricks': {
        'learner': optimizee.trick.Mixed,
        'kwargs': {
            'MLP_Scale': {'learner': optimizee.scaleMLP.ScaleMultilayerPerceptron,
                          'kwargs': {
                              'depth': 1,
                              'activation': 'sigmoid',
                              'is_scaling': 3.0,
                              'data_path': '../dataset/mnist',
                              'batch_size': 128
                          },  # number of hidden layers, activation, is_scaling
                          'coe': 1.0},
            'convex': {'learner': optimizee.trivial.Square,
                       'kwargs': {
                           'is_scaling': 1.0,
                           'dim': 20
                       },
                       'coe': 1.0},
        },
        "lr": 0.001,
    }
}
