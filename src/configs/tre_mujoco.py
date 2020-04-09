from dataclasses import dataclass

@dataclass
class Config:
    game = "Reacher"
    algo = "TRE"
    seed = 0
    device_id = 0

    # Training related
    update_frequency = 1

    max_steps = int(1e6)
    exploration_steps = 25000
    log_interval = 5000
    eval_episodes = 10
    save_interval = int(1e5)
    action_noise_level = 0.1

    # Replay related
    buffer_size = int(1e6)
    batch_size = 256

    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    p_lr = 3e-4
    v_lr = 3e-4
    tau = 0.005

    # TRPO related
    max_kl = 1e-2
    cg_damping = 0.1
    cg_iters = 10
    accept_ratio = 0.1
    max_backtracks = 10
    residual_tol = 1e-10

    # Others
    hidden_size = 256
    ckpt = ""
    log_dir = ""
    play = False
