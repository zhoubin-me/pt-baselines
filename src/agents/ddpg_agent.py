import torch
import torch.nn.functional as F
from copy import deepcopy
from .base_agent import BaseAgent
from src.common.async_replay import AsyncReplayBuffer
from src.common.make_env import make_bullet_env
from src.common.utils import tensor

class DDPGAgent(BaseAgent):
    def __init__(self, cfg):
        super(DDPGAgent, self).__init__(cfg)

        self.envs = make_bullet_env(cfg.game, cfg.log_dir + '/train', seed=cfg.seed)()
        self.test_env = make_bullet_env(cfg.game, cfg.log_dir + '/test', seed=cfg.seed+1)()

        self.target_network = deepcopy(self.network)
        self.actor_optimizer = torch.optim.Adam(self.network.get_policy_params(), lr=cfg.p_lr)
        self.critic_optimizer = torch.optim.Adam(self.network.get_value_params(), lr=cfg.v_lr, weight_decay=cfg.v_w_decay)

        self.max_action = float(self.envs.action_space.high[0])
        self.train_state = None

    def step(self):
        if self.total_steps == 0:
            self.train_state = self.envs.reset()

        if self.total_steps < self.cfg.start_timesteps:
            action = self.envs.action_space.sample()
        else:
            state = tensor(self.train_state).float().to(self.device).unsqueeze(0)
            pi = self.network.p(self.state_normalizer(state))
            action, _ = self.act(pi)
            action = action.squeeze(0).cpu().numpy()


        next_state, reward, done, info = self.envs.step(action)
        self.rollouts.add([self.train_state, action, reward, next_state, done])
        self.train_state = next_state
        self.total_steps += 1

        if 'episode' in info:
            self.train_state = self.envs.reset()
            self.logger.store(TrainEpRet=info['episode']['r'])

    def eval_step(self):
        state = tensor(self.test_state).float().to(self.device).unsqueeze(0)
        pi = self.network.p(self.state_normalizer(state))
        action, _ = self.act(pi)
        return action.squeeze(0).cpu().numpy()

    def update(self):
        if self.total_steps < self.cfg.start_timesteps:
            return

        cfg = self.cfg
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.rollouts.sample()

        target_q = self.target_network.action_value(next_obs_batch, self.target_network.p(next_obs_batch))
        target_q = reward_batch + (1.0 - done_batch) * cfg.gamma * target_q.detach()
        current_q = self.network.action_value(obs_batch, action_batch)
        value_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()


        policy_loss = self.network.action_value(obs_batch, self.network.p(obs_batch)).mean().neg()
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


