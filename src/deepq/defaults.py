class Config:

    game = 'Breakout'
    env_id = '{game}NoFrameskip-v4'

    dueling = True
    double = True
    prioritize = True
    nstep = 3
    noise = True
    noise_std = 0.5

    max_steps = int(5e7)
    discount = 0.99

    batch_size = 32
    adam_lr = 0.0000625
    adam_eps = 0.00015

    memory_size = int(1e6)
    




