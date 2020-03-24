

class RainbowConfig:
    game = 'Breakout'
    seed = 0

    dueling = True
    double = True
    prioritize = True
    noisy = True

    nstep = 3
    noise_std = 0.5
    num_atoms = 51
    v_max = 10
    v_min = -10
    history_length = 4

    sgd_update_frequency = 4
    discount = 0.99
    batch_size = 32
    adam_lr = 0.0000625
    adam_eps = 0.00015

    max_steps = int(5e7)
    max_episode_steps = 108000
    log_interval = 10000
    eval_interval = 100000
    save_interval = 1000000
    eval_episodes = 10

    exploration_steps = 20000
    target_network_update_freq = 8000
    epsilon_steps = int(1e6)
    min_epsilon = 0.01
    test_epsilon = 0.01

    replay_size = int(1e6)
    replay_alpha = 0.5
    replay_beta0 = 0.4

    ckpt = ""
    log_dir = ""
    play = False


class A3CConfig:
    game = 'Pong'
    seed = 0

    num_actors = 16

    nsteps = 20
    discount = 0.99
    adam_lr = 0.0001

    gae_coef = 1.0
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 40

    max_episode_steps = 108000
    max_steps = int(4e7)
    eval_episodes = 10

    ckpt = ""
    log_dir = ""
    play = False


class A2CConfig:
    game = "Pong"
    seed = 0

    num_processes = 16
    use_gae = False

    nsteps = 5
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
    log_interval = 8000
    eval_episodes = 10
    use_lr_decay = False

    ckpt = ""
    log_dir = ""
    play = False


class PPOConfig:
    game = "Pong"
    seed = 1

    num_processes = 8
    use_gae = True

    nsteps = 128
    gamma = 0.99
    lr = 2.5e-4
    eps = 1e-5
    alpha = 0.99
    epoches = 4
    num_mini_batch = 4


    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1

    max_episode_steps = 108000
    max_steps = int(1e7)
    log_interval = nsteps * num_processes * 10
    eval_episodes = 10
    use_lr_decay = True

    ckpt = ""
    log_dir = ""
    play = False