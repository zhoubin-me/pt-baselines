
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)
        nn.init.zeros_(m.bias.data)


class ACNet(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(ACNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ELU(),
            nn.Flatten(),
        )

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        self.fc_v = nn.Linear(256, 1)
        self.fc_pi = nn.Linear(256, action_dim)

        self.apply(init)
        nn.init.orthogonal_(self.fc_pi.weight.data, 0.01)


    def forward(self, x):
        ix, (hx, cx) = x
        phi = self.convs(ix)
        hx, cx = self.lstm(phi, (hx, cx))
        return self.fc_v(hx), self.fc_pi(hx), (hx, cx)

