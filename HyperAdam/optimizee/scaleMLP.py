import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from optimizee.__init__ import Optimizee
from optimizee.modules import ScaleLinear

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# no need to "softmax"
LOSS = nn.CrossEntropyLoss()



class ScaleMultilayerPerceptron(Optimizee):
    def __init__(self, kwargs):
        # depth is the number of hidden layers
        super().__init__()
        for key_ in ['depth', 'is_scaling', 'activation', 'data_path', 'batch_size']:
            if key_ not in kwargs:
                raise KeyError

        self.depth = kwargs['depth']
        self.scale = kwargs['is_scaling']
        self._get_activation(kwargs['activation'])
        self.data_path = kwargs['data_path']
        self.batch_size = kwargs['batch_size']
        self.add_module('linear0', ScaleLinear(784, 20, scale=self.scale))
        for d in range(1, self.depth):
            name = 'linear' + str(d + 1)
            self.add_module(name, ScaleLinear(20, 20, scale=self.scale))
        self.add_module('last_layer', ScaleLinear(20, 10, scale=self.scale))
        self.dim = self.get_dim()

    def get_train_data_loader(self, cycling=True, shuffle=True):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.data_path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=shuffle)
        if cycling is True:
            self.cycling_data()

    def forward(self, inputs=None):
        train_images = inputs.view(-1, 28 * 28)
        x = self.linear0(train_images)
        for d, module_ in enumerate(self.children()):
            if d == 0:
                continue
            x = module_(self.activationFunc(x))
        self.output = x

    def get_loss(self, data_batch=None):
        images, labels = data_batch
        self.forward(inputs=images)
        loss = LOSS(self.output, labels)
        return loss


def main():
    kwargs = {
        'depth': 1,
        'activation': 'sigmoid',
        'is_scaling': True,
        'data_path': '../../dataset/mnist',
        'batch_size': 1
    }
    model = ScaleMultilayerPerceptron(kwargs)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    sh = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.get_train_data_loader()
    for _ in range(3000):
        model.reset_parameters()
        model.get_train_iter()
        model.to(device)
        for epoch in range(2000):
            for i in range(100):
                data = model.next_train_batch(device)
                loss = model.get_loss(data)
                print(epoch, i, loss)
                model.get_grad(loss)
                optimizer.step()


if __name__ == '__main__':
    main()
