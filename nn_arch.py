import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, LogSoftmax

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layer = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(6),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining a 2D convolution layer
            Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(12),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining a 2D convolution layer
            Conv2d(12, 18, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(18),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining a 2D convolution layer
            Conv2d(18, 36, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(36),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(36 * 14 * 14, 3)
        )

        self.softmax = LogSoftmax(dim=1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        x = self.softmax(x)

        return x