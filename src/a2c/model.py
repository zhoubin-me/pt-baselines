import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .distributions import Bernoulli, DiagGaussian



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
        pass


    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        features = self.convs(inputs / 255.0)
        values = self.fc_v(features)
        logits = self.fc_pi(features)
        dist = Categorical(logits=logits)

        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        return values, actions.unsqueeze(-1), action_log_probs.unsqueeze(-1), rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        features = self.convs(inputs / 255.0)
        values = self.fc_v(features)
        return values

    def evaluate_actions(self, inputs, rnn_hxs, masks, actions):
        features = self.convs(inputs / 255.0)
        values = self.fc_v(features)
        logits = self.fc_pi(features)
        dist = Categorical(logits=logits)

        action_log_probs = dist.log_prob(actions.view(-1))
        dist_entropy = dist.entropy().mean()

        return values, action_log_probs.unsqueeze(-1), dist_entropy, rnn_hxs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


