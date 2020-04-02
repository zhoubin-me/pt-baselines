import gym
import pybullet_envs
# import pybulletgym
from gym import wrappers
from src.common.env_wrappers import make_atari, wrap_deepmind, AtariRescale42x42, NormalizedEnv
from src.common.monitor import Monitor
from src.common.vec_env import ShmemVecEnv, VecPyTorch, VecPyTorchFrameStack, DummyVecEnv, VecNormalize

def make_env(game, env_type, **kwargs):
    if env_type == 'atari':
        return make_atari_env(game, **kwargs)
    elif env_type == 'mujoco' or 'bullet':
        return make_bullet_env(game, **kwargs)
        # return make_mujoco_env(game, **kwargs)
    elif env_type == 'bullet':
        return make_bullet_env(game, **kwargs)
    else:
        raise NotImplementedError("Please implement yourself")

def make_bullet_env(game, log_prefix, seed=1234, record_video=False, **kwargs):
    def trunk():
        env = gym.make(f"{game}BulletEnv-v0")
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=True)
        env = wrappers.Monitor(env, f'{log_prefix}', force=True) if record_video else env
        return env
    return trunk

def make_mujoco_env(game, log_prefix, seed=1234, record_video=False, **kwargs):
    def trunk():
        env = gym.make(f"{game}MuJoCoEnv-v0")
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=True)
        env = wrappers.Monitor(env, f'{log_prefix}', force=True) if record_video else env
        return env
    return trunk


def make_atari_env(game,
                   log_prefix,
                   seed=1234,
                   record_video=False,
                   frame_stack=True,
                   episode_life=True,
                   transpose_image=True,
                   clip_rewards=True,
                   allow_early_resets=True):
    def trunk():
        env = make_atari(f'{game}NoFrameskip-v4')
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=allow_early_resets)
        env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=frame_stack, transpose_image=transpose_image)
        env = wrappers.Monitor(env, f'{log_prefix}', force=True) if record_video else env
        return env
    return trunk


def make_a3c_env(game, log_prefix, record_video=False, seed=1234):
    def trunk():
        env = gym.make(f'{game}Deterministic-v4')
        env.seed(seed)
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=True)
        env = wrappers.Monitor(env, f'{log_prefix}', force=True) if record_video else env
        return env
    return trunk


def make_vec_envs(game, log_dir, num_processes, seed, allow_early_resets=True, env_type='atari', record_video=False, **kwargs):

    envs = [
        make_env(game, env_type, log_prefix=f'{log_dir}/rank_{i}',
                 seed=seed+i, frame_stack=False, allow_early_resets=allow_early_resets, **kwargs)
        for i in range(num_processes)
    ]

    envs = ShmemVecEnv(envs, context='fork') if num_processes > 1 else DummyVecEnv(envs)
    envs = VecNormalize(envs) if env_type == 'mujoco' or env_type == 'bullet' else envs

    envs = VecPyTorch(envs)
    envs = VecPyTorchFrameStack(envs, 4) if env_type == 'atari' else envs

    return envs
