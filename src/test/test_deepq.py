import unittest
import gym

from src.common.async_replay import AsyncReplayBuffer
from src.common.atari_wrapper import make_atari, wrap_deepmind


class TestAsyncReplay(unittest.TestCase):

    def test_async(self):
        memory = AsyncReplayBuffer(int(1e6), 32, True, 0.5)

        env = make_atari('BreakoutNoFrameskip-v4', 108000)
        env = wrap_deepmind(env, frame_stack=True)
        s = env.reset()

        while True:
            action = env.action_space.sample()
            s_next, reward, done, _ = env.step(action)
            memory.add((s, action, reward, done, s_next))
            if done:
                break
            else:
                s = s_next

        for _ in range(10):
            o = memory.sample(beta=0.4)
            for x in o:
                print(type(o), o.size())
    
