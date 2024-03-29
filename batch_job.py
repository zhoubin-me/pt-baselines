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
NUM_THREADS = 40
seed = 4

def run_single_config(queue):
    while True:
        config_path, game = queue.get()
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

for cfg in cfgs:
    if 'ddpg' in cfg:
        pass
    else:
        continue

    for i, game in enumerate(_mujoco7):
        if 'ddpg' in cfg or 'rainbow' in cfg:
            os.system(f'python run.py {cfg} --seed {seed} --game {game} --device_id {i} & ')
        else:
            q.put((cfg, game))

q.join()
