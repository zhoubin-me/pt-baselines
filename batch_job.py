import os
import torch
import glob
from src.common.bench import _atari7, _mujoco8


N = torch.cuda.device_count()
cfgs = glob.glob('src/configs/*.py')
_mujoco8.pop(-2)

for cfg in cfgs:
    if 'trpo' in cfg and 'mujoco' in cfg:
        pass
    else:
        continue

    if 'atari' in cfg:
        for i, game in enumerate(_atari7):
            os.system(f'export CUDA_VISIBLE_DEVICES="{i % N}";python -m run {cfg} --game {game} --seed 1')
    elif 'mujoco' in cfg:
        for i, game in enumerate(_mujoco8):
            if not 'Inverted' in game:
                continue
            os.system(f'export CUDA_VISIBLE_DEVICES="{i % N}";python -m run {cfg} --game {game} --seed 1')
