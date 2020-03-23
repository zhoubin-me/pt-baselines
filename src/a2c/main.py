import argparse
import numpy as np

from src.common.utils import set_thread, random_seed, mkdir
from src.a2c.a2c import A2CAgent


class Config:
    game = 'PongNoFrameskip-v4'
    # game = "Pong"
    seed = 0

    num_processes = 16
    recurrent = True

    nsteps = 5
    gamma = 0.99
    rms_lr = 7e-4
    rms_eps = 1e-5
    rms_alpha = 0.99

    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    max_episode_steps = 108000
    max_steps = int(5e7)
    log_interval = 8000
    eval_episodes = 10

    ckpt = ""
    log_dir = ""
    play = False
    device_id = 0

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

    # random_seed(args.seed)
    # set_thread(1)

    mkdir(args.log_dir)
    agent = A2CAgent(args=args)
    # agent.logger.save_config(args)

    if not args.play:
        mkdir(args.ckpt_dir)
        agent.run()
    else:
        agent.load(args.ckpt)
        print(f"Running {args.eval_episodes} episodes")
        returns = agent.eval_episodes()
        print(f"Returns: {returns}")
        print(f"Max: {np.max(returns)}\t Min: {np.min(returns)}\t Std: {np.std(returns)}")

