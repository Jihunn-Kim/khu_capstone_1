import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.quantization import QuantStub, DeQuantStub

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(8, 8, 3),
          nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(8, 8, 3),
          nn.ReLU(True),
        )
        self.fc4 = nn.Linear(8 * 23 * 23, 2)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc4(x)
        x = self.dequant(x)
        return x
  
    def fuse_model(self):
        for m in self.modules():
          if type(m) == nn.Sequential:
              torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)


class NetContainer(nn.Module):
    def __init__(self, input_channels, num_filters, input_dim, output_dim=2):
        super(NetContainer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, num_filters[0], 3),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(num_filters[0], num_filters[1], 3),
          nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(num_filters[1], num_filters[2], 3),
          nn.ReLU(True),
        )
        self.fc4 = nn.Sequential(
          nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc4(x)
        return x

class NetContainerConvBlock(nn.Module):
    def __init__(self, input_channels, num_filters):
        super(NetContainerConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, num_filters[0], 3),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(num_filters[0], num_filters[1], 3),
          nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(num_filters[1], num_filters[2], 3),
          nn.ReLU(True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x