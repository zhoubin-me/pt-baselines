
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
        self.test_state = None
        self.state_normalizer = None
        self.reward_normalizer = None

    def close(self):
        close_obj(self.test_env)

    def save(self, filename):

        torch.save(self.network.state_dict(), '%s.model' % (filename))
        if hasattr(self.test_env, 'ob_rms'):
            with open('%s.stats' % (filename), 'wb') as f:
                pickle.dump({
                    'ob_rms': self.test_env.ob_rms,
                    'ret_rms': self.test_env.ret_rms
                }, f)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        if hasattr(self.test_env, 'ob_rms'):
            with open('%s.stats' % (filename[:-6]), 'rb') as f:
                rms = pickle.load(f)
                self.test_env.ob_rms = rms['ob_rms']
                self.test_env.ret_rms = rms['ret_rms']

    def eval_step(self):
        raise NotImplementedError

    def eval_episode(self):
        env = self.test_env
        if self.test_state is None or self.cfg.algo == 'Rainbow' or self.cfg.algo == 'DDPG':
            self.test_state = env.reset()

        while True:
            action = self.eval_step()
            state, reward, done, info = env.step(action)
            self.test_state = state
            if isinstance(info, list):
                info = info[0]
            if 'episode' in info:
                ret = info['episode']['r']
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        self.network.eval()
        for ep in range(self.cfg.eval_episodes):
            with torch.no_grad():
                total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.network.train()
        return episodic_returns
