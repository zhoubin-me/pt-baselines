

from src.common.utils import set_thread, random_seed
from src.deepq.deepq import RainbowAgent

class Config:
    game = 'Breakout'
    seed = 0

    dueling = True
    double = True
    prioritize = True
    nstep = 3
    noisy = True
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
    exploration_steps = 20000
    target_network_update_freq = 8000
    epsilon_steps = int(1e6)
    min_epsilon = 0.01

    replay_size = int(1e6)
    replay_alpha = 0.5
    replay_beta0 = 0.4



if __name__ == '__main__':
    for k, v in Config.__dict__.items():
        print(k ,v)