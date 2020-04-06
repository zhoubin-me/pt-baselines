from dataclasses import dataclass

@dataclass
class Config:
    game = "Reacher"
    algo = "TD3"
    seed = 0
    device_id = 0

    # Training related
    buffer_size = int(1e6)
    batch_size = 256
    sgd_update_frequency = 1
    policy_update_freq = 2

    max_steps = int(1e6)
    exploration_steps = 25000
    log_interval = 5000
    eval_episodes = 10
    save_interval = int(1e5)


    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    p_lr = 3e-4
    v_lr = 3e-4
    tau = 0.005

    # Others
    ckpt = ""
    log_dir = ""
    play = False
