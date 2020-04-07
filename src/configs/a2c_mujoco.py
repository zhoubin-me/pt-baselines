from dataclasses import dataclass

@dataclass
class Config:
    game = "Reacher"
    algo = 'A2C'
    seed = 0
    device_id = -1

    # Trainng related
    num_processes = 5
    mini_steps = 16
    mini_epoches = 1
    num_mini_batch = 1

    max_steps = int(1e6)
    log_interval = num_processes * mini_steps * 100
    eval_episodes = 10
    save_interval = int(1e5)
    use_lr_decay = True

    # Optimizer related
    optimizer = 'rmsprop'
    gamma = 0.99
    lr = 3e-4
    eps = 1e-5
    alpha = 0.99

    # Loss related
    gae_lambda = 0.95
    entropy_coef = 0
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    # Others
    ckpt = ""
    log_dir = ""
    play = False
