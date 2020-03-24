import torch
from torch.distributions import Categorical

import time
from collections import namedtuple

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from celegans.agents.base_agent import BaseAgent
from celegans.common.utils import make_vec_envs
from celegans.common.model import ACNet
from celegans.common.logger import EpochLogger
from celegans.common.schedule import LinearSchedule
from celegans.common.normalizer import SignNormalizer, ImageNormalizer

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'action_log_probs', 'rewards', 'values', 'masks', 'returns'])


class PPOAgent(BaseAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

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


    def sampler(self, advs):
        cfg = self.cfg
        rollouts = self.rollouts
        batch_size = cfg.num_processes * cfg.nsteps
        mini_batch_size = batch_size // cfg.num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape)[indices]
            action_batch = rollouts.actions.view(-1, 1)[indices]
            action_log_prob_batch = rollouts.action_log_probs.view(-1, 1)[indices]
            value_batch = rollouts.values[:-1].view(-1, 1)[indices]
            mask_batch = rollouts.masks[:-1].view(-1, 1)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]
            adv_batch = advs.view(-1, 1)[indices]
            yield obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, adv_batch



    def update(self):
        cfg = self.cfg
        rollouts = self.rollouts

        advantages = rollouts.returns[:-1] - rollouts.values[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        for e in range(cfg.epoches):
            data_generator = self.sampler(advantages)
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch,  adv_targ = sample

                # Reshape to do in a single forward pass for all steps

                values, pis = self.network(obs_batch)
                dist = Categorical(logits=pis)
                action_log_probs = dist.log_prob(actions_batch.view(-1))
                dist_entropy = dist.entropy().mean()

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param,
                                    1.0 + cfg.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-cfg.clip_param, cfg.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                             value_losses_clipped).mean()


                loss = value_loss * cfg.value_loss_coef + action_loss - dist_entropy * cfg.entropy_coef
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()

        num_updates = cfg.epoches * cfg.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss


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

