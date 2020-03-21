import unittest
import gym
import torch


from src.common.utils import make_vec_env, mkdir
from src.a3c.model import ACNet
from torch.distributions import Categorical



class TestAsyncReplay(unittest.TestCase):

    def test_async(self):
        envs = make_vec_env('Pong', 'log/a3c-test')
        mkdir('log/a3c-test')
        s = envs.reset()
        net = torch.nn.Linear(4 * 84 * 84, envs.action_space.n).cuda()
        print(s.shape)
        for step in range(10000):

            with torch.no_grad():
                logits = net(s.view(s.size(0), -1))
                actions = Categorical(logits=logits).sample()
            s_next, reward, dones, infos = envs.step(actions)
            for info in infos:
                if 'episode' in info:
                    rs = info['episode']['r']
                    print(f"=================={rs}================={step}==={dones}")


