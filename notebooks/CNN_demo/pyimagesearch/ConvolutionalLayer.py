from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import BatchNorm2d
from torch import flatten


class ConvolutionLayer(Module):
    def __init__(self, numChannels, classes) -> None:

        # call the parent constructor
        super(ConvolutionLayer, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(
            in_channels=numChannels, out_channels=16, kernel_size=(5, 5), padding=(2, 2)
        )
        self.batchNorm1 = BatchNorm2d(16)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2)
        )
        self.batchNorm2 = BatchNorm2d(32)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=1568, out_features=500)
        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU => POOL layers
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the input through our second set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the volume, then FC => RELU => FC
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        # apply a softmax to the output
        x = self.logSoftmax(x)

        # return the output
        return x
