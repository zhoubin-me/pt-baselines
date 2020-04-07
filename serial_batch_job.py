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
    exps = [
        'a2c_atari', 'a2c_mujoco',
        'a3c_atari',
    ]

    if not any(map(lambda x: x in cfg, exps)):
        continue

    if 'mujoco' in cfg:
        games = _mujoco7
    else:
        games = _atari7

    for game in games:
        os.system(f'python run.py {cfg} --game {game} --seed {seed}')
