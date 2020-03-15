import torch
import numpy as np
import random
import os
from pathlib import Path

from gym import wrappers
from src.common.atari_wrapper import make_atari, wrap_deepmind
from src.common.monitor import Monitor

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


def make_env(game, log_prefix, record_video=False, seed=1234):
    def trunk():
        env = make_atari(f'{game}NoFrameskip-v4')
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=True)
        env = wrap_deepmind(env, episode_life=not record_video, frame_stack=True)
        if record_video:
            env = wrappers.Monitor(env, f'{log_prefix}', force=True)
        return env
    return trunk()


def random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(np.random.randint(int(1e6)))
