import os
import torch
import glob
import sys
import json
from src.common.bench import _atari7, _mujoco7
from run import main

seed = 1
cfgs = glob.glob('src/configs/*.py')
for cfg in cfgs:

    exps = [
        # 'ppo_atari',
        # 'trpo_atari',
        'ddpg_mujoco',
        # 'td3_mujoco',
        # 'rainbow_atari',
        'sac_mujoco'
    ]

    if not any(map(lambda x: x in cfg, exps)):
        continue

    if 'mujoco' in cfg:
        games = _mujoco7
    else:
        games = _atari7

    for i, game in enumerate(games):
        hold_on = " " if game == games[-1] else "&"
        os.system(f'python run.py {cfg} --game {game} --seed {seed} --device_id {i} {hold_on}')
