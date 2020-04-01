




class A3CConfig:
    game = 'Pong'
    env_type = 'atari'
    seed = 0

    num_actors = 16

    mini_steps = 20
    discount = 0.99
    adam_lr = 0.0001

    gae_coef = 1.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 40

    max_episode_steps = 108000
    max_steps = int(1e7)
    eval_episodes = 10
    save_interval = int(1e6)

    ckpt = ""
    log_dir = ""
    play = False


class A2CConfig:
    game = "Pong"
    env_type = 'atari'
    seed = 0
    sep_body = False

    num_processes = 16
    mini_epoches = 1
    num_mini_batch = 1

    optimizer = 'rmsprop'
    mini_steps = 5
    gamma = 0.99
    lr = 7e-4
    eps = 1e-5
    alpha = 0.99

    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    max_episode_steps = 108000
    max_steps = int(1e7)
    log_interval = mini_steps * num_processes * 100
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = False

    ckpt = ""
    log_dir = ""
    play = False


class PPOConfig:
    game = "Pong"
    env_type = 'atari'
    seed = 0
    sep_body = False

    num_processes = 8

    optimizer = 'adam'
    mini_steps = 128
    gamma = 0.99
    lr = 2.5e-4
    eps = 1e-5
    alpha = 0.99
    mini_epoches = 4
    num_mini_batch = 4


    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    max_episode_steps = 108000
    max_steps = int(1e7)
    log_interval = mini_steps * num_processes * 10
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = True

    ckpt = ""
    log_dir = ""
    play = False

class PPORobotConfig:
    game = "Reacher"
    env_type = 'mujoco'
    seed = 0
    sep_body = False

    num_processes = 1

    optimizer = 'adam'
    mini_steps = 2048
    gamma = 0.99
    lr = 3e-4
    eps = 1e-5
    alpha = 0.99
    mini_epoches = 10
    num_mini_batch = 32

    gae_lambda = 0.95
    entropy_coef = 0
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    max_episode_steps = 108000
    max_steps = int(1e6)
    log_interval = mini_steps * num_processes * 2
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = True

    ckpt = ""
    log_dir = ""
    play = False

class TRPORobotConfig:
    game = "Reacher"
    env_type = 'mujoco'
    sep_body = True
    seed = 0

    num_processes = 1

    optimizer = 'lbfgs'
    lr = 3e-4
    eps = 1e-5
    mini_steps = 15000
    gamma = 0.995
    mini_epoches = 1
    num_mini_batch = 1

    l2_reg = 1e-3
    max_kl = 1e-2
    cg_damping = 0.1
    cg_iters = 10
    accept_ratio = 0.1
    max_backtracks = 10
    residual_tol = 1e-10

    gae_lambda = 0.97
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    max_episode_steps = 108000
    max_steps = int(1e6)
    log_interval = mini_steps * num_processes * 1
    eval_episodes = 10
    save_interval = int(1e6)
    use_lr_decay = False

    ckpt = ""
    log_dir = ""
    play = False
