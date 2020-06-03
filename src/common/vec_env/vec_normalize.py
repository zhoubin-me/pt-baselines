from . import VecEnvWrapper
from .running_mean_std import RunningStat

import numpy as np


class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass

class RewardFilter:
    """
    Incorrect reward normalization [copied from OAI code]
    update return
    divide reward by std(return) without subtracting and adding back mean
    """

    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff / (self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        self.state_filter = ZFilter(Identity(), shape=self.observation_space.shape, clip=clipob) if ob else None
        self.reward_filter = RewardFilter(Identity(), shape=(), gamma=gamma, clip=cliprew) if ret else None

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = self.state_filter(obs)
        rews = self.reward_filter(rews)
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        return self.state_filter(obs, reset=True)
