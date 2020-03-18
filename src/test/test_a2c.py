import unittest
import gym
import torch


from src.common.async_replay import AsyncReplayBuffer
from src.common.utils import make_a3c_env, mkdir
from src.a3c.model import ACNet
from torch.distributions import Categorical

class TestAsyncReplay(unittest.TestCase):

    def test_async(self):

        env = make_a3c_env('Pong', 'log/a3c-test')
        mkdir('log/a3c-test')
        s = env.reset()
        network = ACNet(s.shape[0], env.action_space.n)
        network.cuda()

        hx, cx = torch.zeros(1, 256).cuda(), torch.zeros(1, 256).cuda()
        step = 0
        while True:
            s = torch.from_numpy(s).unsqueeze(0).cuda()
            v, pi, (hx, cx) = network((s, (hx, cx)))
            d = Categorical(logits=pi)
            a = d.sample()
            if step % 100 == 0:
                print("{0:6.2f}, {1}, {2:6.2f}, {3:6.2f}, {4}".format(v.item(), 'xx', hx.mean().item(), cx.mean().item(), a.item()))


            s_next, reward, done, info = env.step(a.item())
            if 'episode' in info:
                rs = info['episode']['r']
                print(f"=================={rs}================={step}")

            s = s_next
            step += 1
            if done:
                s = env.reset()

