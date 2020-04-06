import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Normal
import copy
import time
import numpy as np

from src.common.async_replay import AsyncReplayBuffer
from src.common.make_env import make_bullet_env
from src.common.logger import EpochLogger
from src.common.model import DDPGMLP, TD3MLP
from src.common.utils import close_obj
from .base_agent import BaseAgent
from .async_actor import AsyncActor


class DDPGActor(AsyncActor):
    def __init__(self, cfg, lock, device_id):
        super(DDPGActor, self).__init__(cfg)
        self.lock = lock
        self.device_id = device_id
        self.start()

    def _set_up(self):
        cfg = self.cfg
        self._device = torch.device(f'cuda:{self.device_id}') if self.device_id >= 0 else torch.device('cpu')
        self._env = make_bullet_env(cfg.game, cfg.log_dir + '/train', seed=cfg.seed)()
        # self._action_high = self._env.action_space.high[0]

    def _transition(self):
        if self._state is None:
            self._state = self._env.reset()


        if self._total_steps < self.cfg.exploration_steps:
            action = self._env.action_space.sample()
        else:
            state = torch.from_numpy(self._state).float().to(self._device).unsqueeze(0)
            pi = self._network.p(state)
            dist = Normal(pi, self._network.p_log_std.expand_as(pi).exp())
            action = dist.sample()
            # action = action.clamp(-self._action_high, self._action_high)
            action = action.squeeze(0).cpu().numpy()

        next_state, reward, done, info = self._env.step(action)
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        self._state = next_state
        if done:
            self._state = self._env.reset()
        return entry


class DDPGAgent(BaseAgent):
    def __init__(self, cfg):
        super(DDPGAgent, self).__init__(cfg)
        self.lock = mp.Lock()
        self.device = torch.device(f'cuda:{cfg.device_id}') if cfg.device_id >= 0 else torch.device('cpu')
        self.actor = DDPGActor(cfg, self.lock, cfg.device_id)
        self.test_env = make_bullet_env(cfg.game, cfg.log_dir + '/test', seed=cfg.seed + 1)()
        self.action_high = self.test_env.action_space.high[0]

        self.logger = EpochLogger(cfg.log_dir, exp_name=cfg.algo)
        self.replay = AsyncReplayBuffer(
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            device_id=cfg.device_id
        )

        if cfg.algo == 'DDPG':
            self.network = DDPGMLP(self.test_env.observation_space.shape[0], self.test_env.action_space.shape[0]).to(self.device)
        elif cfg.algo == 'TD3':
            self.network = TD3MLP(self.test_env.observation_space.shape[0], self.test_env.action_space.shape[0]).to(self.device)
        else:
            raise NotImplementedError

        self.network.train()
        self.network.share_memory()
        self.target_network = copy.deepcopy(self.network)

        self.actor.set_network(self.network)

        self.actor_optimizer = torch.optim.Adam(self.network.get_policy_params(), lr=cfg.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.get_value_params(), lr=cfg.v_lr)

        self.total_steps = 0
        self.max_action = float(self.test_env.action_space.high[0])

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self):
        state = torch.from_numpy(self.test_state).float().to(self.device).unsqueeze(0)
        pi = self.network.p(state)
        dist = Normal(pi, self.network.p_log_std.expand_as(pi).exp())
        action = dist.sample()
        # action = action.clamp(-self.action_high, self.action_high)
        return action.squeeze(0).cpu().numpy()

    def step(self):
        cfg = self.cfg

        ## Environment Step
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.total_steps += 1
            experiences.append([state, action, reward, next_state, done])
            if isinstance(info, dict):
                if 'episode' in info:
                    self.logger.store(TrainEpRet=info['episode']['r'])
        self.replay.add_batch(experiences)


    def update(self):
        cfg = self.cfg
        if self.total_steps < cfg.exploration_steps:
            return

        ## Upate
        cfg = self.cfg
        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        states = states.float()
        next_states = next_states.float()
        actions = actions.float()
        terminals = terminals.float().view(-1, 1)
        rewards = rewards.float().view(-1, 1)

        target_q = self.target_network.action_value(next_states, self.target_network.p(next_states))
        target_q = rewards + (1.0 - terminals) * cfg.gamma * target_q.detach()
        current_q = self.network.action_value(states, actions)
        value_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = self.network.action_value(states, self.network.p(states)).mean().neg()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        kwargs = {
            'Loss': 0,
            'VLoss': value_loss.item(),
            'PLoss': policy_loss.item(),
            'Entropy': None,
        }
        self.logger.store(**kwargs)

    def run(self):

        logger = self.logger
        cfg = self.cfg

        t0 = time.time()
        logger.store(TrainEpRet=0, Loss=0)
        last_epoch = -1

        while self.total_steps < cfg.max_steps:
            self.step()
            self.update()

            if self.total_steps % cfg.log_interval == 0:
                logger.log_tabular('TotalEnvInteracts', self.total_steps)
                logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
                logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
                logger.log_tabular('TrainEpRet', with_min_and_max=True)
                logger.log_tabular('Loss', average_only=True)
                logger.log_tabular('RemHrs',
                                   (cfg.max_steps - self.total_steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
                t0 = time.time()
                logger.dump_tabular(self.total_steps)

            epoch = self.total_steps // self.cfg.save_interval
            if epoch > last_epoch:
                last_epoch = epoch
                self.save(f'{cfg.ckpt_dir}/{self.total_steps:08d}')
                test_returns = self.eval_episodes()
                test_tabular = {
                    "Epoch": self.total_steps // cfg.save_interval,
                    "Steps": self.total_steps,
                    "NumOfEp": len(test_returns),
                    "AverageTestEpRet": np.mean(test_returns),
                    "StdTestEpRet": np.std(test_returns),
                    "MaxTestEpRet": np.max(test_returns),
                    "MinTestEpRet": np.min(test_returns)}
                logger.dump_test(test_tabular)

        self.close()







