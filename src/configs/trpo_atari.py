from dataclasses import dataclass

@dataclass
class Config:
    game = "Pong"
    algo = "TRPO"
    seed = 0
    device_id = 0

    # Training related
    num_processes = 1
    mini_steps = 2048
    mini_epoches = 1
    num_mini_batch = 32

    max_steps = int(1e7)
    log_interval = 10000
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = True

    # Optimizer related
    optimizer = 'adam'
    lr = 2.5e-4
    eps = 1e-5
    gamma = 0.99

    # TRPO related
    max_kl = 1e-2
    cg_damping = 0.1
    cg_iters = 10
    accept_ratio = 0.1
    max_backtracks = 10
    residual_tol = 1e-10

    # Loss related
    gae_lambda = 0.97
    entropy_coef = 0.01
    max_grad_norm = 0.5
    clip_param = 0.1

    # Others
    norm_env = False
    ckpt = ""
    log_dir = ""
    play = False
