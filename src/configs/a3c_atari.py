from dataclasses import dataclass

@dataclass
class Config:
    game = 'Pong'
    algo = 'A3C'
    seed = 0

    # Number of parallel threads
    num_actors = 16

    # Optimizer related
    mini_steps = 20 # Accumulate samples before update
    discount = 0.99
    adam_lr = 0.0001

    # Loss Hypers
    gae_coef = 1.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 40

    # Training Related
    max_steps = int(1e7)
    eval_episodes = 10
    save_interval = int(1e6)

    ckpt = ""
    log_dir = ""
    play = False