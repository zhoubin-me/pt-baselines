from dataclasses import dataclass

@dataclass
class Config:
    game = 'Breakout'
    algo = 'Rainbow'
    env_type = 'atari'
    seed = 0

    # Rainbow related
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
    discount = 0.99
    target_network_update_freq = 8000

    # Optimizer related
    sgd_update_frequency = 4
    batch_size = 32
    adam_lr = 0.0000625
    adam_eps = 0.00015

    # Training related
    max_steps = int(1e7)
    log_interval = 10000
    eval_interval = int(1e5)
    save_interval = int(1e6)
    eval_episodes = 10

    # Exploration related
    exploration_steps = 20000
    epsilon_steps = int(1e6)
    min_epsilon = 0.01
    test_epsilon = 0.01

    # Replay memory related
    replay_size = int(1e6)
    replay_alpha = 0.5
    replay_beta0 = 0.4

    # Others
    ckpt = ""
    log_dir = ""
    play = False