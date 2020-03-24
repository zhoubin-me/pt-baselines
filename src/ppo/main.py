import argparse
import numpy as np

from src.common.utils import set_thread, random_seed, mkdir
from src.agents.ppo_agent import PPOAgent


class Config:
    game = "Pong"
    seed = 1234

    num_processes = 8
    recurrent = True

    nsteps = 128
    gamma = 0.99
    lr = 2.5e-4
    eps = 1e-5
    alpha = 0.99
    epoches = 4
    num_mini_batch = 4


    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    max_episode_steps = 108000
    max_steps = int(5e7)
    log_interval = nsteps * num_processes * 10
    eval_episodes = 10
    use_lr_decay = True

    ckpt = ""
    log_dir = ""
    play = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Hyperparameters')
    for k, v in Config.__dict__.items():
        if not k.startswith('_'):
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()
    print(args)

    if len(args.log_dir) == 0:
        args.log_dir = f'log/ppo-{args.game}-{args.seed}/'
        args.ckpt_dir = f'ckpt/ppo-{args.game}-{args.seed}/'

    if len(args.ckpt) > 0:
        args.log_dir = f'log/ppo-{args.game}-{args.seed}-eval/'

    random_seed(args.seed)
    set_thread(1)

    mkdir(args.log_dir)
    agent = PPOAgent(cfg=args)
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

