import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(2, 4, 3),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(4, 8, 3),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc4 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc4(x)
        return x


IN_DIM      = 1     # byte
STATE_DIM   = 16
FEATURE_DIM = 16
class OneNet(nn.Module):
    def __init__(self):
        super(OneNet, self).__init__()
        
        # transform the given packet into a tensor which is in a good feature space
        self.feature_layer = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, FEATURE_DIM),
            nn.ReLU()
        )

        # generates the current state 's'
        self.f = nn.Sequential(
            nn.Linear(STATE_DIM + FEATURE_DIM, STATE_DIM),
            nn.ReLU(),
            nn.Linear(STATE_DIM, STATE_DIM),
            nn.ReLU()
        )

        # check whether the given packet is malicious
        self.g = nn.Sequential(
            nn.Linear(STATE_DIM + FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Flatten()
        )

    def forward_state(self, x, s):
        x   = self.feature_layer(x)
        x   = torch.cat((x, s), 2)
        s2 = self.f(x)
        return s2
    
    def forward(self, x, s):
        x   = self.feature_layer(x)
        x   = torch.cat((x, s), 2)
        x2 = self.g(x)
        return x2
