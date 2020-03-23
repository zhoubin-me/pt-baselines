import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from src.common.monitor import Monitor
from src.common.atari_wrapper import make_atari, wrap_deepmind
from src.common.vec_env import VecEnvWrapper
from src.common.vec_env.dummy_vec_env import DummyVecEnv
from src.common.vec_env.vec_pytorch import VecPyTorch, VecPyTorchFrameStack
from src.common.vec_env.shmem_vec_env import ShmemVecEnv
from src.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_



def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env = make_atari(env_id)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, transpose_image=True)
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    envs = ShmemVecEnv(envs, context='fork')
    envs = VecPyTorch(envs, device)
    envs = VecPyTorchFrameStack(envs, 4, device)
    return envs


