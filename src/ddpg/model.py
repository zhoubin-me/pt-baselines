import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ACNet(nn.Module):
    def __init__(self, state_dim, action_dim):

        self.body = nn.Sequential(
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear()
        )

        super()
        pass