import torch
from torch.distributions import Categorical

from src.a2c.agent import A2CAgent
from src.common.schedule import LinearSchedule
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)
        self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.lr, eps=cfg.eps)

        if cfg.use_lr_decay:
            scheduler = LinearSchedule(1.0, 0, cfg.max_steps // (cfg.num_processes * cfg.nsteps))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler)
        else:
            self.lr_scheduler = None

    def sample(self):
        cfg = self.cfg
        rollouts = self.rollouts
        batch_size = cfg.num_processes * cfg.nsteps
        mini_batch_size = batch_size // cfg.num_mini_batch

        adv = self.rollouts.returns[:-1] - self.rollouts.values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape)[indices]
            action_batch = rollouts.actions.view(-1, 1)[indices]
            action_log_prob_batch = rollouts.action_log_probs.view(-1, 1)[indices]
            value_batch = rollouts.values[:-1].view(-1, 1)[indices]
            mask_batch = rollouts.masks[:-1].view(-1, 1)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]
            adv_batch = adv.view(-1, 1)[indices]
            yield obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, adv_batch



    def update(self):
        cfg = self.cfg

        vloss, ploss, entropies, losses, counter = 0, 0, 0, 0, 0
        for epoch in range(cfg.epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, adv_batch = batch_data

                vs, pis = self.network(obs_batch)
                dist = Categorical(logits=pis)
                log_probs = dist.log_prob(action_batch.view(-1))
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - action_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * adv_batch
                policy_loss = torch.min(surr1, surr2).mean().neg()

                value_loss = (vs - return_batch).pow(2)

                vs_clipped = value_batch + (vs - value_batch).clamp(-cfg.clip_param, cfg.clip_param)
                vs_loss_clipped = (vs_clipped - return_batch).pow(2)

                value_loss = 0.5 * torch.max(value_loss, vs_loss_clipped).mean()


                loss = value_loss * cfg.value_loss_coef + policy_loss - entropy * cfg.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                vloss += value_loss.item()
                ploss += policy_loss.item()
                entropies += entropy.item()
                losses += loss.item()
                counter += 1

        vloss /= counter
        ploss /= counter
        entropies /= counter
        losses /= counter

        return vloss, ploss, entropies, losses




