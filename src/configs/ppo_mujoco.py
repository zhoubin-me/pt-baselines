from dataclasses import dataclass

@dataclass
class Config:
    game = "Reacher"
    algo = "PPO"
    env_type = 'mujoco'
    seed = 0
    sep_body = False
    
    # Training related
    num_processes = 1
    mini_steps = 2048
    mini_epoches = 10
    num_mini_batch = 32
    
    max_steps = int(1e6)
    log_interval = 4096
    eval_episodes = 10
    save_interval = mini_steps * num_processes * 4
    use_lr_decay = True
    
    
    # Optimizer related
    optimizer = 'adam'
    gamma = 0.99
    lr = 3e-4
    eps = 1e-5
    alpha = 0.99
    
    
    # Loss related
    gae_lambda = 0.95
    entropy_coef = 0
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    clip_param = 0.1
    
    
    # Others
    ckpt = ""
    log_dir = ""
    play = False