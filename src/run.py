import numpy as np
import argparse
from src.common.utils import random_seed, mkdir, set_thread
from src.agents import *
from src.config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters Settings')
    parser.add_argument('algo', type=str, choices=['Rainbow', 'A2C', 'A3C', 'PPO', 'TRPO'])
    parser.add_argument('env_type', type=str, choices=['atari', 'mujoco'])
    args = parser.parse_known_args()
    if args[0].env_type == 'atari':
        cfg = eval(f'{args[0].algo}Config')
    else:
        cfg = eval(f'{args[0].algo}RobotConfig')

    for k, v in cfg.__dict__.items():
        if not k.startswith('_'):
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()

    if len(args.log_dir) == 0:
        args.log_dir = f'log/{args.algo}-{args.game}-{args.seed}/'
        args.ckpt_dir = f'ckpt/{args.algo}-{args.game}-{args.seed}/'

    if len(args.ckpt) > 0:
        args.log_dir = f'log/{args.algo}-{args.game}-{args.seed}-eval/'

    if args.seed <= 0:
        args.seed = 0
    else:
        random_seed(args.seed)

    if args.algo == 'A3C' or args.algo == 'Rainbow':
        set_thread(1)

    mkdir(args.log_dir)
    agent = eval(f'{args.algo}Agent(cfg=args)')
    agent.logger.save_config(vars(args))

    if not args.play:
        mkdir(args.ckpt_dir)
        agent.run()
    else:
        agent.load(args.ckpt)
        print(f"Running {args.eval_episodes} episodes")
        returns = agent.eval_episodes()
        print(f"Returns: {returns}")
        print(f"Max: {np.max(returns)}\t Min: {np.min(returns)}\t Std: {np.std(returns)}")
