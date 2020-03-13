
import argparse
import numpy as np

from src.common.utils import set_thread, random_seed
from src.deepq.deepq import RainbowAgent


class Config:
    game = 'Breakout'
    seed = 0

    dueling = True
    double = True
    prioritize = True
    noisy = True

    nstep = 3
    noise_std = 0.5
    num_atoms = 51
    v_max = 10
    v_min = -10
    history_length = 4

    sgd_update_frequency = 4
    discount = 0.99
    batch_size = 32
    adam_lr = 0.0000625
    adam_eps = 0.00015

    max_steps = int(5e7)
    log_interval = 10000
    eval_interval = 100000
    save_interval = 1000000
    eval_episodes = 10

    exploration_steps = 20000
    target_network_update_freq = 8000
    epsilon_steps = int(1e6)
    min_epsilon = 0.01
    test_epsilon = 0.01

    replay_size = int(1e6)
    replay_alpha = 0.5
    replay_beta0 = 0.4

    ckpt = ""
    log_dir = ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rainbow Hyperparameters')
    print("===========Deepq Hyperparameters Setting=======================")
    for k, v in Config.__dict__.items():
        if not k.startswith('_'):
            parser.add_argument(f'--{k}', type=type(v), default=v)
            print(f"||\t{k}\t\t\t\t||\t\t\t\t{v}")
    print("===========Deepq Hyperparameters Setting=======================")
    args = parser.parse_args()

    if len(args.log_dir) == 0:
        args.log_dir = f'log/rainbow-{args.game}-{args.seed}/'

    if len(args.ckpt) > 0:
        args.log_dir = f'log/rainbow-{args.game}-{args.seed}-eval/'

    agent = RainbowAgent(cfg=args)

    if len(args.ckpt) == 0:
        agent.run()
    else:
        agent.load(args.ckpt)
        print(f"Running {args.eval_episodes} episodes")
        returns = agent.eval_episodes()
        print(f"Returns: {returns}")
        print(f"Max: {np.max(returns)}\t Min: {np.min(returns)}\t Std: {np.std(returns)}")

