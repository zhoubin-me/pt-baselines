import torch
import torch.nn as nn


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


    def forward(self, x):
        ix, (hx, cx) = x
        phi = self.convs(ix)
        hx, cx = self.lstm(phi, (hx, cx))
        return self.fc_v(hx), self.fc_pi(hx), (hx, cx)


if __name__ == '__main__':
    net = ACNet(4, 4)

    xs = torch.randn(32, 4, 42, 42), (torch.zeros(32, 256), torch.zeros(32, 256))
    print(net(xs))