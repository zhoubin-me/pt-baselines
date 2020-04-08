import os
import torch
import glob
from multiprocessing import Process, JoinableQueue
import sys
import json
from src.common.bench import _atari7, _mujoco7
from run import main

cfgs = glob.glob('src/configs/*.py')
q = JoinableQueue()
NUM_THREADS = 50

def run_single_config(queue):
    while True:
        config_path, game, seed = queue.get()
        try:
            main(cfg=config_path, game=game, seed=seed)
        except Exception as e:
            print("ERROR", e)
            raise e
        queue.task_done()

for i in range(NUM_THREADS):
    worker = Process(target=run_single_config, args=(q,))
    worker.daemon = True
    worker.start()

for seed in [1, 2, 3]:
    for cfg in cfgs:
        exps = ['trpo_mujoco', 'ppo_mujoco']
        if any(map(lambda x: x in cfg, exps)):
            pass
        else:
            continue

        for game in _mujoco7:
            q.put((cfg, game, seed))
q.join()


