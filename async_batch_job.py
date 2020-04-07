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
    if 'mujoco' in cfg or 'rainbow' in cfg:
        pass
    else:
        continue

    algos = ['td3', 'ddpg', 'rainbow']
    if any(map(lambda x: x in cfg, algos)):
        pass
    else:
        continue

    if 'rainbow' in cfg:
        games = _atari7
    else:
        games = _mujoco7

    for game in games:
        hold_on = " " if game == games[-1] else "&"
        os.system(f'python run.py {cfg} --game {game} --seed {seed} {hold_on}')
