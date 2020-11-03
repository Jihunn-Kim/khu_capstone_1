import torch.nn as nn
import torch.nn.functional as F
import torch
import const

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.f1 = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.ReLU(True),
        )
        self.f2 = nn.Sequential(
          nn.Conv2d(2, 4, 3),
          nn.ReLU(True),
        )
        self.f3 = nn.Sequential(
          nn.Conv2d(4, 8, 3),
          nn.ReLU(True),
        )
        self.f4 = nn.Sequential(
          nn.Linear(8 * 23 * 23, 2),
        )

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = torch.flatten(x, 1)
        x = self.f4(x)
        return x