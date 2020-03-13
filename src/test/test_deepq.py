import unittest
import gym

from src.common.async_replay import AsyncReplayBuffer
from src.common.atari_wrapper import make_atari, wrap_deepmind
from src.deepq.model import C51Net

class TestAsyncReplay(unittest.TestCase):

    def test_async(self):
        memory = AsyncReplayBuffer(int(1e6), 32, True, 0.5)

        env = make_atari('BreakoutNoFrameskip-v4', 108000)
        env = wrap_deepmind(env, frame_stack=True)
        s = env.reset()

        network = C51Net(env.action_space.n, 51, noisy=True, duel=True)
        network.cuda()
        network.reset_noise(0.5)

        while True:
            action = env.action_space.sample()
            s_next, reward, done, _ = env.step(action)
            memory.add((s, action, reward, done, s_next))
            if done:
                break
            else:
                s = s_next

        for _ in range(10):
            s, a, r, d, s_next, _, _ = memory.sample(beta=0.4)
            prob, log_prob = network(s)
            print(prob.shape, log_prob.shape)

            


