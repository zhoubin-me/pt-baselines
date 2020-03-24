
import torch
import unittest

class Test(unittest.TestCase):

    def test_lr(self):

        from delegans.common.model import ACNet
        from delegans.common.schedule import LinearSchedule
        net = ACNet(4, 4)

        optimizer = torch.optim.Adam(net.parameters(), 2.5e-4)
        scheduler = LinearSchedule(1, 0, int(1e7) // (8 * 128))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e7) // (8 * 128))
        print(int(1e7) // (8 * 128))
        for step in range(int(1e7) // (8 * 128)):

            v, p = net(torch.randn(32, 4, 84, 84))
            y = torch.randn(32)
            loss = (v - y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            print(step, lr_scheduler.get_lr())
