from dataclasses import dataclass

@dataclass
class Config:
    game = "Reacher"
    algo = "DDPG"
    seed = 0
    device_id = 0

    # Training related
    num_processes = 1
    buffer_size = int(1e6)
    batch_size = 256

    max_steps = int(1e6)
    log_interval = 5000
    eval_episodes = 10
    save_interval = int(1e5)


    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    p_lr = 3e-4
    v_lr = 3e-4

    # Others
    ckpt = ""
    log_dir = ""
    play = False
