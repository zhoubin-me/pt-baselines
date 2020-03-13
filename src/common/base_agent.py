
import torch
import pickle
from src.common.utils import close_obj

# Copied from ShangtongZhang/DeepRL
# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/BaseAgent.py

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.task_ind = 0
        self.network = None
        self.task = None

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            if 'episode' in info[0]:
                ret = info[0]['episode']['r']
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        self.network.eval()
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.network.train()
        return episodic_returns