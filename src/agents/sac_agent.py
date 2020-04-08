import torch
import torch.nn.functional as F
import numpy as np

from .ddpg_agent import DDPGAgent


class SACAgent(DDPGAgent):
    def __init__(self, cfg):
        super(SACAgent, self).__init__(cfg)

        self.target_entropy = torch.tensor(np.prod(self.test_env.action_space.shape)).to(self.device).float().neg()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.p_lr)

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
            next_actions, next_entropies, _ = self.network.act(next_states)
            target_q1, target_q2 = self.target_network.action_value(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) + self.log_alpha.exp() * next_entropies
            target_q = rewards + (1.0 - terminals) * (cfg.gamma ** cfg.nsteps) * target_q.detach()

        current_q1, current_q2 = self.network.action_value(states, actions)
        value_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        sampled_action, entropy, _ = self.network.act(states)
        q1, q2 = self.network.action_value(states, sampled_action)
        q = torch.min(q1, q2)
        policy_loss = (q + self.log_alpha.exp() * entropy).mean().neg()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        entropy_loss = (self.log_alpha * (self.target_entropy - entropy).detach()).mean().neg()
        self.alpha_optim.zero_grad()
        entropy_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.network.get_value_params(), self.target_network.get_value_params()):
            target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)


        kwargs = {
            "PLoss": policy_loss,
            "Entropy": entropy_loss,
            "VLoss": value_loss
        }
        self.logger.store(**kwargs)




