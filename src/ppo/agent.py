import torch
from torch.distributions import Categorical

from src.a2c.agent import A2CAgent
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



class PPOAgent(A2CAgent):
    def __init__(self, cfg):
        super(A2CAgent, self).__init__(cfg)
        self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.adam_lr, eps=cfg.adam_eps)


    def sample(self):
        cfg = self.cfg
        rollouts = self.rollouts
        batch_size = cfg.num_processes * cfg.nsteps
        for epoch in range(cfg.epoches):
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), cfg.mini_batch_size, drop_last=True)
            for indices in sampler:
                obs_batch = rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape)[indices]
                action_batch = rollouts.actions.view(-1, 1)[indices]
                value_batch = rollouts.values[:-1].view(-1, 1)[indices]
                mask_batch = rollouts.masks[:-1].view(-1, 1)[indices]
                returns_batch = rollouts.returns[:-1].view(-1, 1)[indices]


    def update(self):
        cfg = self.cfg

        advs = self.rollouts.returns[:-1] - self.rollouts.values[:-1]
        advs = (advs - advs.mean()) / (advs.std() + 1e-5)




        vs, pis = self.network(self.rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape))
        vs = vs.view(cfg.nsteps, cfg.num_processes, 1)
        dist = Categorical(logits=pis)
        log_probs = dist.log_prob(self.rollouts.actions.view(-1))
        log_probs = log_probs.view(cfg.nsteps, cfg.num_processes, 1)

        advs = self.rollouts.returns[:-1] - vs
        value_loss = advs.pow(2).mean()

        action_loss = 0 - (advs.detach() * log_probs + dist.entropy() * cfg.entropy_coef).mean()

        loss = value_loss * cfg.value_loss_coef + action_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        self.logger.store(Loss=loss.item())

        self.rollouts.obs[0].copy_(self.rollouts.obs[-1])
        self.rollouts.masks[0].copy_(self.rollouts.masks[-1])




