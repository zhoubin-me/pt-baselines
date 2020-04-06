import torch
import numpy as np
import random
import os
from pathlib import Path


def share_rms(master_env, slave_env):
    slave_env.ob_rms = master_env.ob_rms
    slave_env.ret_rms = slave_env.ret_rms

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, dtype=torch.float32).cuda()
    return x

def explained_variance_1d(ypred, y):
  """
  Var[ypred - y] / var[y].
  https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
  """
  vary = torch.var(y)
  return None if vary == 0 else 1 - torch.var(y - ypred) / vary


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


