import numpy as np
import argparse
from src.common.utils import set_thread, random_seed, mkdir
from src.a2c.agent import A2CAgent
from src.a3c.agent import A3CAgent
from src.ppo.agent import PPOAgent
from src.deepq.agent import RainbowAgent
from .config import RainbowConfig, A2CConfig, A3CConfig, PPOConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters Settings')
    parser.add_argument('algo', type=str, choices=['Rainbow', 'A2C', 'A3C', 'PPO'])
    args = parser.parse_known_args()


    cfg = eval(f'{args[0].algo}Config')
    for k, v in cfg.__dict__.items():
        if not k.startswith('_'):
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()

    if len(args.log_dir) == 0:
        args.log_dir = f'log/a2c-{args.game}-{args.seed}/'
        args.ckpt_dir = f'ckpt/a2c-{args.game}-{args.seed}/'

    if len(args.ckpt) > 0:
        args.log_dir = f'log/a2c-{args.game}-{args.seed}-eval/'

    if args.seed is not None:
        random_seed(args.seed)

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