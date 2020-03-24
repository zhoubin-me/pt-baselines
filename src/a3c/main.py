import argparse
import numpy as np

from src.common.utils import set_thread, random_seed, mkdir
from src.agents.a3c_agent import A3CAgent


class Config:
    game = 'Pong'
    seed = 0

    num_actors = 16

    nsteps = 20
    discount = 0.99
    adam_lr = 0.0001

    gae_coef = 1.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 40

    max_episode_steps = 108000
    max_steps = int(4e7)
    eval_episodes = 10

    ckpt = ""
    log_dir = ""
    play = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rainbow Hyperparameters')
    for k, v in Config.__dict__.items():
        if not k.startswith('_'):
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()
    print(args)

    if len(args.log_dir) == 0:
        args.log_dir = f'log/a3c-{args.game}-{args.seed}/'
        args.ckpt_dir = f'ckpt/a3c-{args.game}-{args.seed}/'

    if len(args.ckpt) > 0:
        args.log_dir = f'log/a3c-{args.game}-{args.seed}-eval/'

    random_seed(args.seed)
    set_thread(1)

    mkdir(args.log_dir)
    agent = A3CAgent(cfg=args)
    agent.logger.save_config(args)

    if not args.play:
        mkdir(args.ckpt_dir)
        agent.run()
    else:
        agent.load(args.ckpt)
        print(f"Running {args.eval_episodes} episodes")
        returns = agent.eval_episodes()
        print(f"Returns: {returns}")
        print(f"Max: {np.max(returns)}\t Min: {np.min(returns)}\t Std: {np.std(returns)}")

