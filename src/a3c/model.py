import torch
import torch.nn as nn

import numpy as np

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


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


        self.apply(weights_init)
        self.fc_pi.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.fc_pi.bias.data.fill_(0)
        self.fc_v.weight.data = normalized_columns_initializer(
            self.fc_v.weight.data, 1.0)
        self.fc_v.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()


    def forward(self, x):
        ix, (hx, cx) = x
        phi = self.convs(ix)
        hx, cx = self.lstm(phi, (hx, cx))
        return self.fc_v(hx), self.fc_pi(hx), (hx, cx)


if __name__ == '__main__':
    net = ACNet(4, 4)

    xs = torch.randn(32, 4, 42, 42), (torch.zeros(32, 256), torch.zeros(32, 256))
    print(net(xs))