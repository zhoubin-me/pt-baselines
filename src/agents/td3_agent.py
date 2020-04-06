
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from .ddpg_agent import DDPGAgent



class TD3Agent(DDPGAgent):
    def __init__(self, cfg):
        super(TD3Agent, self).__init__(cfg)

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


        ## Upate
        if self.total_steps > cfg.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = states.float()
            next_states = next_states.float()
            actions = actions.float()
            terminals = terminals.float().view(-1, 1)
            rewards = rewards.float().view(-1, 1)


            with torch.no_grad():
                dist = Normal(self.target_network.p(next_states), self.network.p_log_std.expand_as(actions).exp())
                next_actions = dist.sample()
                next_actions = next_actions.clamp(-self.action_high, self.action_high)
                target_q1, target_q2 = self.network.action_value(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1.0 - terminals) * cfg.gamma * target_q.detach()

            current_q1, current_q2 = self.network.action_value(states, actions)
            value_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.logger.store(VLoss = value_loss.item())
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            if self.total_steps % cfg.update_freq == 0:
                policy_loss = self.network.v1(torch.cat([states, self.network.p(states)], dim=1)).mean().neg()
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()
                self.logger.store(VLoss=policy_loss.item())

                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

