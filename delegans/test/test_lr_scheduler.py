
import torch
import unittest
from torch.distributions import Categorical
class Test(unittest.TestCase):

    def test_lr(self):

        from delegans.common.model import ACNet
        from delegans.common.schedule import LinearSchedule
        from delegans.common.utils import make_vec_envs
        envs= make_vec_envs('Pong', log_dir='./log', num_processes=16, seed=1)
        net = ACNet(4, envs.action_space.n).cuda()

        num_updates = int(1e7) // (8 * 128)
        optimizer = torch.optim.Adam(net.parameters(), 2.5e-4)
        scheduler = LinearSchedule(1, 0, int(1e7) // (8 * 128))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / num_updates))

        obs = envs.reset()
        print(int(1e7) // (8 * 128))
        for step in range(5):

            v, p = net(obs)
            dist = Categorical(logits=p)
            actions = dist.sample()
            print(actions)
            print(isinstance(actions))

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            #
            # lr_scheduler.step()
            # print(step, lr_scheduler.get_lr())
