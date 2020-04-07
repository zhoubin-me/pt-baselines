import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain

def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)

class TD3MLP(nn.Module):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(TD3MLP, self).__init__()
        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.v2 = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )


        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim), nn.Tanh()
        )

        self.apply(lambda m: init(m, np.sqrt(2)))

    def act(self, x):
        return self.p(x) * self.max_action

    def action_value(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.v(x), self.v2(x)

    def get_policy_params(self):
        return self.p.parameters()

    def get_value_params(self):
        return chain(self.v.parameters(), self.v2.parameters())

class DDPGMLP(nn.Module):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(DDPGMLP, self).__init__()

        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs + action_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim), nn.Tanh()
        )

        self.apply(lambda m: init(m, np.sqrt(2)))

    def act(self, x):
        return self.p(x) * self.max_action

    def action_value(self, state, action):
        return self.v(torch.cat([state, action], dim=1))

    def get_policy_params(self):
        return self.p.parameters()

    def get_value_params(self):
        return self.v.parameters()

class SepBodyConv(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(SepBodyConv, self).__init__()
        self.v = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.p = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.v[-1].apply(lambda m: init(m, 1.0))
        self.p[-1].apply(lambda m: init(m, 0.01))

    def forward(self, x):
        values = self.v(x)
        logits = self.p(x)
        return values, logits


    def get_policy_params(self):
        return self.p.parameters()

    def get_value_params(self):
        return self.v.parameters()


class SepBodyMLP(nn.Module):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(SepBodyMLP, self).__init__()

        self.max_action = max_action
        self.v = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.p = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_dim), nn.Tanh()
        )

        self.apply(lambda m: init(m, np.sqrt(2)))
        self.p_log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)

    def forward(self, x):
        values = self.v(x)
        logits = self.p(x) * self.max_action
        return values, logits

    def get_policy_params(self):
        return chain(self.p.parameters(), iter([self.p_log_std]))

    def get_value_params(self):
        return self.v.parameters()


class MLPNet(nn.Module):
    def __init__(self, num_inputs, action_dim, max_action, hidden_size=256):
        super(MLPNet, self).__init__()

        self.max_action = max_action
        self.mlps = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )

        self.v = nn.Linear(hidden_size, 1)
        self.p = nn.Linear(hidden_size, action_dim)

        self.apply(lambda m: init(m, np.sqrt(2)))
        self.p_log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)

    def forward(self, x):
        features = self.mlps(x)
        values = self.v(features)
        logits = self.p(features).tanh() * self.max_action
        return values, logits


class ConvNet(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(ConvNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512), nn.ReLU())

        self.v = nn.Linear(512, 1)
        self.p = nn.Linear(512, action_dim)

        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.p.apply(lambda m: init(m, 0.01))
        self.v.apply(lambda m: init(m, 1.0))

    def forward(self, x):
        features = self.convs(x)
        values = self.v(features)
        logits = self.p(features)
        return values, logits


class LightACNet(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(LightACNet, self).__init__()

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

        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.fc_pi.apply(lambda m: init(m, 0.01))
        self.fc_v.apply(lambda m: init(m, 1.0))
        self.lstm.apply(lambda m: init(m, 1.0))


    def forward(self, x):
        ix, (hx, cx) = x
        phi = self.convs(ix)
        hx, cx = self.lstm(phi, (hx, cx))
        return self.fc_v(hx), self.fc_pi(hx), (hx, cx)


class NoisyLinear(nn.Module):
    def __init__(self, in_size, out_size, sigma=0.5):
        super(NoisyLinear, self).__init__()
        self.linear_mu = nn.Linear(in_size, out_size)
        self.linear_sigma = nn.Linear(in_size, out_size)

        self.register_buffer('noise_w', torch.zeros_like(self.linear_mu.weight))
        self.register_buffer('noise_b', torch.zeros_like(self.linear_mu.bias))

        self.sigma = sigma

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            x = F.linear(x,
                         self.linear_mu.weight + self.linear_sigma.weight * self.noise_w,
                         self.linear_mu.bias + self.linear_sigma.bias * self.noise_b)
        else:
            x = self.linear_mu(x)
        return x

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.linear_mu.weight.size(1))
        self.linear_mu.weight.data.uniform_(-stdv, stdv)
        self.linear_mu.bias.data.uniform_(-stdv, stdv)

        self.linear_sigma.weight.data.fill_(self.sigma * stdv)
        self.linear_sigma.bias.data.fill_(self.sigma * stdv)

    def reset_noise(self, std=None):
        self.noise_w.data.normal_()
        self.noise_b.data.normal_()

class C51Net(nn.Module):
    def __init__(self, action_dim, num_atoms, noisy=False, noise_std=0.5, duel=False, in_channels=4):
        super(C51Net, self).__init__()
        self.num_atoms = num_atoms
        self.action_dim = action_dim
        self.noise_std = noise_std
        FC = NoisyLinear if noisy else nn.Linear

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_q = nn.Sequential(
            FC(64 * 7 * 7, 512), nn.ReLU(),
            FC(512, num_atoms * action_dim)
        )

        if duel:
            self.fc_v = nn.Sequential(
                FC(64 * 7 * 7, 512), nn.ReLU(),
                FC(512, num_atoms)
            )
        else:
            self.fc_v = None

    def forward(self, x):
        phi = self.convs(x)
        q = self.fc_q(phi).view(-1, self.action_dim, self.num_atoms)

        if self.fc_v is not None:
            v = self.fc_v(phi)
            q = v.view(-1, 1, self.num_atoms) + q - q.mean(dim=1, keepdim=True)

        prob = F.softmax(q, dim=-1)
        log_prob = F.log_softmax(q, dim=-1)
        return prob, log_prob

    def reset_noise(self, std=None):
        if std is None: std = self.noise_std
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise(std)

