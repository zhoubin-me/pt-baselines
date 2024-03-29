
class Config:
    game = "Reacher"
    algo = 'DDPG'
    seed = 0
    device_id = 0
    num_processes = 1

    # Trainng related
    max_steps = int(1e6)
    exploration_steps = 25000
    sgd_update_frequency = 1
    batch_size = 256
    buffer_size = int(1e6)

    log_interval = 1e4
    eval_episodes = 10
    save_interval = int(1e5)
    use_lr_decay = False

    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    lr = 3e-4
    eps = 1e-5

    p_lr = 1e-4
    v_lr = 1e-3
    v_w_decay = 1e-2
    tau = 0.005

    # Others
    norm_env = False
    ckpt = ""
    log_dir = ""
    play = False
