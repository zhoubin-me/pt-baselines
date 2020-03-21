import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



def init_module(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        nn.init.zeros_(m.bias.data)



class ACNet(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(ACNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), Flatten(),
            nn.Linear(32 * 7 * 7, 512), nn.ReLU())

        self.fc_v = nn.Linear(512, 1)
        self.fc_pi = nn.Linear(512, action_dim)

        self.apply(init_module)
        nn.init.orthogonal_(self.fc_pi.weight.data, gain=0.01)


    def forward(self, x):
        features = self.convs(x)
        values = self.fc_v(features)
        logits = self.fc_pi(features)
        return values, logits






