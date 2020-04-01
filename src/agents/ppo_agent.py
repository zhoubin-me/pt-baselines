import torch
from torch.distributions import Categorical, Normal
from gym.spaces import Discrete, Box

from .a2c_agent import A2CAgent
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

    def sample(self):
        cfg = self.cfg
        rollouts = self.rollouts
        batch_size = cfg.num_processes * cfg.nsteps
        mini_batch_size = batch_size // cfg.num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape)[indices]
            action_batch = rollouts.actions.view(-1, self.action_store_dim)[indices]
            action_log_prob_batch = rollouts.action_log_probs.view(-1, 1)[indices]
            value_batch = rollouts.values[:-1].view(-1, 1)[indices]
            mask_batch = rollouts.masks[:-1].view(-1, 1)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]
            gae_batch = rollouts.gaes[:-1].view(-1, 1)[indices]
            yield obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch

    def update(self):
        cfg = self.cfg

        for epoch in range(cfg.epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch = batch_data
                adv_batch = (gae_batch - gae_batch.mean()) / (gae_batch.std() + 1e-5)

                vs, pis = self.network(obs_batch)

                if isinstance(self.envs.action_space, Discrete):
                    dist = Categorical(logits=pis)
                    action_log_probs = dist.log_prob(action_batch.view(-1)).unsqueeze(-1)
                    entropy = dist.entropy().mean()

                elif isinstance(self.envs.action_space, Box):
                    dist = Normal(pis, self.network.p_log_std.expand_as(pis).exp())
                    action_log_probs = dist.log_prob(action_batch).sum(-1, keepdim=True)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    raise NotImplementedError('No such action space')


                ratio = torch.exp(action_log_probs - action_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_batch
                policy_loss = torch.min(surr1, surr2).mean().neg()

                value_loss = (vs - gae_batch - value_batch).pow(2)
                vs_clipped = value_batch + (vs - value_batch).clamp(-cfg.clip_param, cfg.clip_param)
                vs_loss_clipped = (vs_clipped - gae_batch - value_batch).pow(2)

                value_loss = 0.5 * torch.max(value_loss, vs_loss_clipped).mean()

                loss = value_loss * cfg.value_loss_coef + policy_loss - entropy * cfg.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                kwargs = {
                    'Loss': loss.item(),
                    'VLoss': value_loss.item(),
                    'PLoss': policy_loss.item(),
                    'Entropy': entropy.item(),
                }
                self.logger.store(**kwargs)








