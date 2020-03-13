import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise(std_init)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self, std):
        self.std_init = std
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)



class C51Net(nn.Module):
    def __init__(self, action_dim, num_atoms, noisy=False, noise_std=0.5, duel=False, in_channels=4):
        super(C51Net, self).__init__()
        self.num_atoms = num_atoms
        self.action_dim = action_dim
        self.noise_std = noise_std
        M = NoisyLinear if noisy else nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc = M(64 * 7 * 7, 512)
        self.A = M(512, num_atoms * action_dim)
        if duel:
            self.V = M(512, num_atoms)
        else:
            self.V = None

    def forward(self, x):
        phi = self.fc(self.convs(x))
        a = self.A(phi).view(-1, self.action_dim, self.num_atoms)
        if self.V is not None:
            v = self.v(phi)
            q = v + a - a.mean(dim=1, keepdim=True)
        else:
            q = a
        prob = F.softmax(q, dim=-1)
        log_prob = F.log_softmax(q, dim=-1)
        return prob, log_prob
    
    def reset_noise(self, std=None):
        if std is None: std = self.noise_std
        for name, m in self.named_children():
            if isinstance(m, NoisyLinear):
                m.reset_noise(std)