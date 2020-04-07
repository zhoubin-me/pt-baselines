import os
import torch
import glob
import sys
import json
from src.common.bench import _atari7, _mujoco7
from run import main

cfgs = glob.glob('src/configs/*.py')

seed = 1
for cfg in cfgs:
    if 'a2c' in cfg:
        pass
    else:
        continue

    for game in _mujoco7:
        os.system(f'python run.py {cfg} --game {game} --seed {seed} & ')
