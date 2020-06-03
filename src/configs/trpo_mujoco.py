from dataclasses import dataclass

@dataclass
class Config:
    game = "Reacher"
    algo = "TRPO"
    seed = 0
    device_id = -1

    # Training related
    num_processes = 1
    mini_steps = 2048
    mini_epoches = 10
    num_mini_batch = 32

    max_steps = int(1e6)
    log_interval = 4096
    eval_episodes = 10
    save_interval = int(1e5)
    use_lr_decay = True

    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    lr_v = 3e-4
    lr_p = 3e-4
    eps = 1e-5

    # TRPO related
    max_kl = 1e-2
    cg_damping = 0.1
    cg_iters = 10
    accept_ratio = 0.1
    max_backtracks = 10
    residual_tol = 1e-10
    fisher_frac = 0.125

    # Loss related
    gae_lambda = 0.95
    entropy_coef = 0.0
    max_grad_norm = 0.5
    clip_param = 0.2

    # Others
    hidden_size = 64
    ckpt = ""
    log_dir = ""
    play = False
