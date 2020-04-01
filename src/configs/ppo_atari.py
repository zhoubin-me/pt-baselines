from dataclasses import dataclass

@dataclass
class Config:
    game = "Pong"
    algo = "PPO"
    env_type = 'atari'
    seed = 0
    sep_body = False

    # Training related
    num_processes = 8
    mini_epoches = 4
    num_mini_batch = 4
    mini_steps = 128

    max_steps = int(1e7)
    log_interval = num_processes * mini_steps * 10
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = True

    # Optimization related
    optimizer = 'adam'
    gamma = 0.99
    lr = 2.5e-4
    eps = 1e-5
    alpha = 0.99

    # Loss related
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    # Others
    ckpt = ""
    log_dir = ""
    play = False
