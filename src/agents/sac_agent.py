import torch
import torch.nn.functional as F
import numpy as np

from .ddpg_agent import DDPGAgent
from torch.distributions import Normal

class SACAgent(DDPGAgent):
    def __init__(self, cfg):
        super(SACAgent, self).__init__(cfg)

        self.target_entropy = -np.prod(self.test_env.action_space.shape)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.p_lr)

    def sample_action(self, action_mean, action_std):
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_probs =  dist.log_prob(action) - torch.log(1 - action.pow(2) + 1e-5)
        entropy = log_probs.sum(-1, keepdim=True).neg()
        return entropy, action.tanh() * self.action_high

    def update(self):
        cfg = self.cfg

        experiences = self.replay.sample()
        states, actions, rewards, next_states, terminals = experiences
        states = states.float()
        next_states = next_states.float()
        actions = actions.float()
        terminals = terminals.float().view(-1, 1)
        rewards = rewards.float().view(-1, 1)


        with torch.no_grad():
            next_action, next_entropy = self.sample_action(*self.network.act(next_states))
            target_q1, target_q2 = self.target_network.action_value(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) + self.alpha * next_entropy
            target_q = rewards + (1.0 - terminals) * (cfg.gamma ** cfg.nsteps) * target_q.detach()

        current_q1, current_q2 = self.network.action_value(states, actions)
        value_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        current_action, current_entropy = self.sample_action(*self.network.act(states))

        q1, q2 = self.network.action_value(states, current_action)
        q = torch.min(q1, q2)
        policy_loss = (q + self.alpha * current_entropy).mean().neg()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        entropy_loss = (self.log_alpha * (self.target_entropy - current_entropy).detach()).mean().neg()
        self.alpha_optim.zero_grad()
        entropy_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.network.get_value_params(), self.target_network.get_value_params()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)


        self.logger.store(PLoss=policy_loss.item())
        self.logger.store(Entropy=entropy_loss.item())
        self.logger.store(VLoss=value_loss.item())




