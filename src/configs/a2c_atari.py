from dataclasses import dataclass

@dataclass
class Config:
    game = "Pong"
    algo = 'A2C'
    seed = 0
    device_id = 0

    # Trainng related
    num_processes = 16
    mini_steps = 5
    mini_epoches = 1
    num_mini_batch = 1

    max_steps = int(1e7)
    log_interval = num_processes * mini_steps * 100
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = False

    # Optimizer related
    optimizer = 'rmsprop'
    gamma = 0.99
    lr = 7e-4
    eps = 1e-5
    alpha = 0.99

    # Loss related
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    # Others
    ckpt = ""
    log_dir = ""
    play = False
