import torch.nn as nn
import torch


STATE_DIM   = 8 * 32
FEATURE_DIM = 32
class OneNet(nn.Module):
    def __init__(self, packet_num):
        super(OneNet, self).__init__()

        IN_DIM = 8 * packet_num # byte
        
        # transform the given packet into a tensor which is in a good feature space
        self.feature_layer = nn.Sequential(
            nn.Linear(IN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(32, FEATURE_DIM),
            nn.ReLU()
        )

        # generates the current state 's'
        self.f = nn.Sequential(
            nn.Linear(STATE_DIM + FEATURE_DIM, STATE_DIM),
            nn.ReLU(),
            nn.Linear(STATE_DIM, STATE_DIM),
            nn.ReLU(),
            nn.Linear(STATE_DIM, STATE_DIM),
            nn.ReLU()
        )

        # check whether the given packet is malicious
        self.g = nn.Sequential(
            nn.Linear(STATE_DIM + FEATURE_DIM, STATE_DIM + FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(STATE_DIM + FEATURE_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Flatten()
        )

    def forward(self, x, s):
        x   = self.feature_layer(x)
        x   = torch.cat((x, s), 1)

        s2  = self.f(x)
        x2  = self.g(x)

        return x2, s2
