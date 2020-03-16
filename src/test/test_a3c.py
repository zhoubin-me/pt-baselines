import unittest
import gym
import torch


from src.common.async_replay import AsyncReplayBuffer
from src.common.utils import make_a3c_env, mkdir
from src.a3c.model import ACNet
from torch.distributions import Categorical

class TestAsyncReplay(unittest.TestCase):

    def test_async(self):

        env = make_a3c_env('Breakout', 'log/a3c-test')
        mkdir('log/a3c-test')
        s = env.reset()
        network = ACNet(s.shape[0], env.action_space.n)
        network.cuda()

        hx, cx = torch.zeros(1, 256).cuda(), torch.zeros(1, 256).cuda()
        rs = 0
        while True:
            s = torch.from_numpy(s).unsqueeze(0).cuda()
            v, pi, (hx, cx) = network((s, (hx, cx)))

            d = Categorical(logits=pi)
            a = d.sample()

            s_next, reward, done, info = env.step(a.item())
            rs += reward
            s = s_next
            if done:
                s = env.reset()
                print(rs)
                print(v.shape, pi.shape, hx.shape, cx.shape)
                rs = 0

