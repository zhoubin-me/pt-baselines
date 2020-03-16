
import torch
import pickle
import numpy as np
from src.common.utils import close_obj

# Copied from ShangtongZhang/DeepRL
# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/BaseAgent.py

class BaseAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.network = None
        self.test_env = None
        self.state_normalizer = None

    def close(self):
        close_obj(self.test_env)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename[:-6]), 'rb') as f:
            self.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.test_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            if isinstance(info, dict):
                if 'episode' in info:
                    ret = info['episode']['r']
                    break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        self.network.eval()
        for ep in range(self.cfg.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.network.train()
        return episodic_returns
