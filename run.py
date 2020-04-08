import numpy as np
import argparse
import git
from importlib.machinery import SourceFileLoader
from src.common.utils import random_seed, mkdir, set_thread
from src.agents import *

def main(**kwargs):
    parser = argparse.ArgumentParser(description='Hyperparameters Settings')
    if 'cfg' in kwargs:
        cfg = SourceFileLoader('', kwargs['cfg']).load_module().Config
    else:
        parser.add_argument('cfg', type=str, help="Check configurations under src/configs/*.py")
        args = parser.parse_known_args()[0]
        cfg = SourceFileLoader('', args.cfg).load_module().Config

    cfg.sha = git.Repo().head.object.hexsha

    for k, v in cfg.__dict__.items():
        if not k.startswith('_'):
            if k in kwargs:
                parser.add_argument(f'--{k}', type=type(v), default=kwargs[k])
            else:
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

if __name__ == '__main__':
    main()
