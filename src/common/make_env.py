import gym
import pybullet_envs
from gym import wrappers
from src.common.env_wrappers import make_atari, wrap_deepmind, AtariRescale42x42, NormalizedEnv, TimeLimit
from src.common.monitor import Monitor
from src.common.vec_env import ShmemVecEnv, VecPyTorch, VecPyTorchFrameStack

def make_env(game, env_type, **kwargs):
    if env_type == 'atari':
        return make_atari_env(game, **kwargs)
    elif env_type == 'mujoco':
        return make_robot_env(game, **kwargs)
    else:
        raise NotImplementedError("Please implement yourself")

def make_robot_env(game,
                   log_prefix,
                   record_video=False,
                   seed=1234,
                   allow_early_resets=True, **kwargs):
    def trunk():
        env = gym.make(f"{game}BulletEnv-v0")
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=allow_early_resets)
        if record_video:
            env = wrappers.Monitor(env, f'{log_prefix}', force=True)
        return env
    return trunk

def make_atari_env(game,
                   log_prefix,
                   record_video=False,
                   max_episode_steps=108000,
                   seed=1234,
                   frame_stack=True,
                   episode_life=True,
                   transpose_image=True,
                   clip_rewards=True,
                   allow_early_resets=True):
    def trunk():
        env = make_atari(f'{game}NoFrameskip-v4', max_episode_steps)
        env.seed(seed)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=allow_early_resets)
        env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=frame_stack, transpose_image=transpose_image)
        if record_video:
            env = wrappers.Monitor(env, f'{log_prefix}', force=True)
        return env
    return trunk


def make_a3c_env(game, log_prefix, record_video=False, max_episode_steps=108000, seed=1234):
    def trunk():
        env = gym.make(f'{game}Deterministic-v4')
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        env.seed(seed)
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
        env = Monitor(env=env, filename=log_prefix, allow_early_resets=True)
        if record_video:
            env = wrappers.Monitor(env, f'{log_prefix}', force=True)
        return env
    return trunk


def make_vec_envs(game, log_dir, num_processes, seed, allow_early_resets=False, env_type='atari'):

    envs = [
        make_env(game, env_type, log_prefix=f'{log_dir}/rank_{i}', seed=seed+i, frame_stack=False, allow_early_resets=allow_early_resets)
        for i in range(num_processes)
    ]

    envs = ShmemVecEnv(envs, context='fork')
    envs = VecPyTorch(envs)
    envs = VecPyTorchFrameStack(envs, 4)
    return envs