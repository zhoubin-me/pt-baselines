
import argparse
import torch.multiprocessing as mp
import numpy as np

from src.common.utils import set_thread, random_seed, make_a3c_env
from src.a3c.a3c import train, test
from src.a3c.model import ACNet
import os

class Config:
    game = 'Pong'
    seed = 0

    num_processes = 16
    num_steps = 20

    steps_per_transit = 20
    gamma = 0.99
    lr = 0.0001

    gae_lambda = 1.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 50

    max_episode_length = 1000000

    ckpt = ""
    log_dir = ""
    play = False

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    parser = argparse.ArgumentParser(description='Rainbow Hyperparameters')
    for k, v in Config.__dict__.items():
        if not k.startswith('_'):
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()
    print(args)


    random_seed(args.seed)
    set_thread(1)

    env = make_a3c_env(args.game)
    shared_model = ACNet(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    optimizer = None

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

