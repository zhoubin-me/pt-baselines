import torch
import numpy as np
import random
import os
import gym
from pathlib import Path

from gym import wrappers
from delegans.common.atari_wrapper import make_atari, wrap_deepmind, AtariRescale42x42, NormalizedEnv, TimeLimit
from delegans.common.monitor import Monitor
from delegans.common.vec_env import ShmemVecEnv, VecPyTorch, VecPyTorchFrameStack


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, dtype=torch.float32).cuda()
    return x

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

def set_thread(n):
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    torch.set_num_threads(n)


def random_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(np.random.randint(int(1e6)))
    torch.cuda.manual_seed_all(seed)


def make_deepq_env(game,
                   log_prefix,
                   record_video=False,
                   max_episode_steps=108000,
                   seed=1234,
                   frame_stack=True,
                   episode_life=True,
                   transpose_image=True,
                   clip_rewards=True,
                   allow_early_resets=True):
    def trunk():
        env = make_atari(f'{game}NoFrameskip-v4', max_episode_steps)
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=allow_early_resets)
        env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=frame_stack, transpose_image=transpose_image)
        if record_video:
            env = wrappers.Monitor(env, f'{log_prefix}', force=True)
        return env
    return trunk


def make_a3c_env(game, log_prefix, record_video=False, max_episode_steps=108000, seed=1234):
    def trunk():
        env = gym.make(f'{game}Deterministic-v4')
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        env.seed(seed)
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=True)
        if record_video:
            env = wrappers.Monitor(env, f'{log_prefix}', force=True)
        return env
    return trunk



def make_vec_envs(game, log_dir, num_processes, seed, allow_early_resets=False):

    envs = [
        make_deepq_env(game, log_prefix=f'{log_dir}/rank_{i}', seed=seed+i, frame_stack=False, allow_early_resets=allow_early_resets)
        for i in range(num_processes)
    ]

    envs = ShmemVecEnv(envs, context='fork')
    envs = VecPyTorch(envs)
    envs = VecPyTorchFrameStack(envs, 4)
    return envs
