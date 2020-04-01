from dataclasses import dataclass

@dataclass
class Config:
    game = "Pong"
    algo = "TRPO"
    env_type = 'atari'
    sep_body = True
    seed = 0

    # Training related
    num_processes = 8
    mini_steps = 1024
    mini_epoches = 1
    num_mini_batch = 1

    max_steps = int(1e7)
    log_interval = mini_steps * num_processes
    eval_episodes = 10
    save_interval = mini_steps * num_processes * 10
    use_lr_decay = False

    # Optimizer related
    optimizer = 'adam'
    lr = 2.5e-4
    eps = 1e-5
    gamma = 0.995

    # TRPO related
    max_kl = 1e-3
    cg_damping = 1e-3
    cg_iters = 10
    accept_ratio = 0.1
    max_backtracks = 10
    residual_tol = 1e-10

    # Loss related
    l2_reg = 1e-3
    gae_lambda = 0.97
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    # Others
    ckpt = ""
    log_dir = ""
    play = False
