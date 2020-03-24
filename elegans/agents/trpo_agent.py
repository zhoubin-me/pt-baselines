import torch
from torch.distributions import Categorical

import time
from collections import namedtuple

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from elegans.agents.base_agent import BaseAgent
from elegans.common.utils import make_vec_envs
from elegans.common.model import ACNet
from elegans.common.logger import EpochLogger
from elegans.common.schedule import LinearSchedule
from elegans.common.normalizer import SignNormalizer, ImageNormalizer

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'action_log_probs', 'rewards', 'values', 'masks', 'returns'])


class TRPOAgent(BaseAgent):
    def __init__(self, cfg):
        super(TRPOAgent, self).__init__(cfg)
        self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.lr, eps=cfg.eps)

        self.envs = make_vec_envs(cfg.game, seed=cfg.seed, num_processes=cfg.num_processes, log_dir=cfg.log_dir, allow_early_resets=False)

        self.network = ACNet(4, self.envs.action_space.n).cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.lr, eps=cfg.eps)

        if cfg.use_lr_decay:
            scheduler = LinearSchedule(1, 0, cfg.max_steps // (cfg.num_processes * cfg.nsteps))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler)
        else:
            self.lr_scheduler = None

        self.logger = EpochLogger(cfg.log_dir, exp_name=self.__class__.__name__)
        self.reward_normalizer = SignNormalizer()
        self.state_normalizer = ImageNormalizer()

        self.rollouts = Rollouts(
            obs = torch.zeros(cfg.nsteps + 1, cfg.num_processes,  * self.envs.observation_space.shape).cuda(),
            actions = torch.zeros(cfg.nsteps, cfg.num_processes, 1).cuda(),
            action_log_probs = torch.zeros(cfg.nsteps, cfg.num_processes, 1).cuda(),
            values = torch.zeros(cfg.nsteps + 1, cfg.num_processes, 1).cuda(),
            rewards = torch.zeros(cfg.nsteps, cfg.num_processes, 1).cuda(),
            masks = torch.zeros(cfg.nsteps + 1, cfg.num_processes, 1).cuda(),
            returns = torch.zeros(cfg.nsteps + 1, cfg.num_processes, 1).cuda()
        )

        self.total_steps = 0

    def sampler(self):
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



    def step(self):
        cfg = self.cfg
        with torch.no_grad():
            for step in range(cfg.nsteps):
                v, pi = self.network(self.rollouts.obs[step])
                dist = Categorical(logits=pi)
                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)
                states, rewards, dones, infos = self.envs.step(actions)
                self.total_steps += cfg.num_processes

                self.rollouts.masks[step + 1].copy_(1 - dones)
                self.rollouts.actions[step].copy_(actions.unsqueeze(-1))
                self.rollouts.values[step].copy_(v)
                self.rollouts.action_log_probs[step].copy_(action_log_probs.unsqueeze(-1))
                self.rollouts.rewards[step].copy_(rewards)
                self.rollouts.obs[step + 1].copy_(self.state_normalizer(states))

                for info in infos:
                    if 'episode' in info:
                        self.logger.store(TrainEpRet=info['episode']['r'])

            # Compute R and GAE
            v_next, _, = self.network(self.rollouts.obs[-1])
            self.rollouts.values[-1].copy_(v_next)
            gae = 0
            for step in reversed(range(cfg.nsteps)):
                delta = self.rollouts.rewards[step] + cfg.gamma * self.rollouts.values[step + 1] * self.rollouts.masks[
                    step + 1] - self.rollouts.values[step]
                gae = delta + cfg.gamma * cfg.gae_lambda * self.rollouts.masks[step + 1] * gae
                self.rollouts.returns[step].copy_(gae + self.rollouts.values[step])

    def update(self):
        cfg = self.cfg

        vloss, ploss, entropies, losses, counter = 0, 0, 0, 0, 0
        for epoch in range(cfg.epoches):
            sampler = self.sampler()
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


    def run(self):
        cfg = self.cfg
        logger = self.logger
        logger.store(TrainEpRet=0, VLoss=0, PLoss=0, Entropy=0)
        t0 = time.time()

        states = self.envs.reset()
        self.rollouts.obs[0].copy_(self.state_normalizer(states))
        while self.total_steps < cfg.max_steps:
            # Sample experiences
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.step()
            vloss, ploss, entropy, loss = self.update()
            logger.store(VLoss=vloss)
            logger.store(PLoss=ploss)
            logger.store(Entropy=entropy)
            logger.store(Loss=loss)
            if self.total_steps % cfg.log_interval == 0:
                logger.log_tabular('TotalEnvInteracts', self.total_steps)
                logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
                logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
                logger.log_tabular('TrainEpRet', with_min_and_max=True)
                logger.log_tabular('Loss', average_only=True)
                logger.log_tabular('VLoss', average_only=True)
                logger.log_tabular('PLoss', average_only=True)
                logger.log_tabular('Entropy', average_only=True)
                logger.log_tabular('RemHrs',
                                   (cfg.max_steps - self.total_steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
                t0 = time.time()
                logger.dump_tabular(self.total_steps)

