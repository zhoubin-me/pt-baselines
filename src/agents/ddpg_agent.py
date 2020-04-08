import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Normal
import copy
import time
import numpy as np

from src.common.replay_buffer import ReplayBuffer
from src.common.make_env import make_bullet_env
from src.common.logger import EpochLogger
from src.common.model import DDPGMLP, TD3MLP, SACMLP
from src.common.utils import tensor
from .base_agent import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(self, cfg):
        super(DDPGAgent, self).__init__(cfg)
        self.lock = mp.Lock()
        self.device = torch.device(f'cuda:{cfg.device_id}') if cfg.device_id >= 0 else torch.device('cpu')

        self.env = make_bullet_env(cfg.game, cfg.log_dir + '/train', seed=cfg.seed)()
        self.test_env = make_bullet_env(cfg.game, cfg.log_dir + '/test', seed=cfg.seed + 1)()
        self.action_high = self.test_env.action_space.high[0]

        self.logger = EpochLogger(cfg.log_dir, exp_name=cfg.algo)
        self.replay = ReplayBuffer(size=cfg.buffer_size)

        if cfg.algo == 'DDPG':
            NET = DDPGMLP
        elif cfg.algo == 'TD3':
            NET = TD3MLP
        elif cfg.algo == 'SAC':
            NET = SACMLP
        else:
            raise NotImplementedError

        self.network = NET(self.test_env.observation_space.shape[0], self.test_env.action_space.shape[0],
                           self.action_high, cfg.hidden_size).to(self.device)
        self.network.train()
        self.target_network = copy.deepcopy(self.network)

        self.actor_optimizer = torch.optim.Adam(self.network.get_policy_params(), lr=cfg.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.get_value_params(), lr=cfg.v_lr)

        self.total_steps = 0
        self.noise_std = torch.tensor(self.cfg.action_noise_level * self.action_high).to(self.device)

    def eval_step(self):
        state = tensor(self.test_state).float().to(self.device).unsqueeze(0)
        if self.cfg.algo == 'SAC':
            _, _, action = self.network.act(state)
        else:
            action = self.network.act(state)
        return action.squeeze(0).cpu().numpy()

    def step(self):
        cfg = self.cfg

        ## Environment Step
        if self.total_steps == 0:
            self.state = self.env.reset()

        if self.total_steps < self.cfg.exploration_steps:
            action = self.env.action_space.sample()
        else:
            state = tensor(self.state).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                if self.cfg.algo == 'SAC':
                    action, _, _ = self.network.act(state)
                    action = action.squeeze(0).cpu().numpy()
                else:
                    action_mean = self.network.act(state)
                    dist = Normal(action_mean, self.noise_std.expand_as(action_mean))
                    action = dist.sample().clamp(-self.action_high, self.action_high).squeeze(0).cpu().numpy()

        next_state, reward, done, info = self.env.step(action)
        self.total_steps += 1
        self.replay.add(self.state, action, reward, next_state, int(done))

        if isinstance(info, dict):
            if 'episode' in info:
                self.logger.store(TrainEpRet=info['episode']['r'])

        self.state = next_state
        if done:
            self.state = self.env.reset()

        if self.total_steps > cfg.exploration_steps:
            experiences = self.replay.sample(cfg.batch_size)
            states, actions, rewards, next_states, terminals = map(lambda x: tensor(x).to(self.device).float(), experiences)

            terminals = terminals.float().view(-1, 1)
            rewards = rewards.float().view(-1, 1)
            self.update(states, actions, rewards, next_states, terminals)

    def update(self, *args):
        states, actions, rewards, next_states, terminals = args

        cfg = self.cfg
        with torch.no_grad():
            target_q = self.target_network.action_value(next_states, self.target_network.p(next_states))
            target_q = rewards + (1.0 - terminals) * cfg.gamma * target_q.detach()

        current_q = self.network.action_value(states, actions)
        value_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = self.network.action_value(states, self.network.act(states)).mean().neg()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        self.logger.store(
            VLoss = value_loss,
            PLoss = policy_loss
        )

    def run(self):

        logger = self.logger
        cfg = self.cfg

        t0 = time.time()
        logger.store(TrainEpRet=0, Loss=0)
        last_epoch = -1

        while self.total_steps < cfg.max_steps:
            self.step()

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







