import torch
from torch.distributions import Categorical

import time
from collections import namedtuple

from .base_agent import BaseAgent
from src.common.utils import make_vec_envs
from src.common.model import ACNet
from src.common.logger import EpochLogger
from src.common.normalizer import SignNormalizer, ImageNormalizer

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'action_log_probs', 'rewards', 'values', 'masks', 'returns'])

class A2CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A2CAgent, self).__init__(cfg)

        self.envs = make_vec_envs(cfg.game, seed=cfg.seed, num_processes=cfg.num_processes, log_dir=cfg.log_dir, allow_early_resets=False)

        self.network = ACNet(4, self.envs.action_space.n).cuda()

        if cfg.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.network.parameters(), cfg.lr, eps=cfg.eps, alpha=cfg.alpha)
        elif cfg.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.lr, eps=cfg.eps)
        else:
            raise NotImplementedError(f'No such optimizer {cfg.optimizer}')

        if cfg.use_lr_decay:
            scheduler = lambda step : 1 - step / (cfg.max_steps / cfg.num_processes * cfg.nsteps)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler)
        else:
            self.lr_scheduler = None

        self.logger = EpochLogger(cfg.log_dir, exp_name=cfg.algo)
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

            if cfg.use_gae:
                self.rollouts.values[-1].copy_(v_next)
                gae = 0
                for step in reversed(range(cfg.nsteps)):
                    delta = self.rollouts.rewards[step] + cfg.gamma * self.rollouts.values[step + 1] * self.rollouts.masks[step + 1] - self.rollouts.values[step]
                    gae = delta + cfg.gamma * cfg.gae_lambda * self.rollouts.masks[step + 1] * gae
                    self.rollouts.returns[step].copy_(gae + self.rollouts.values[step])

            else:
                self.rollouts.returns[-1].copy_(v_next)
                for step in reversed(range(cfg.nsteps)):
                    self.rollouts.returns[step] = self.rollouts.returns[step + 1] * cfg.gamma * self.rollouts.masks[step + 1] + self.rollouts.rewards[step]


    def update(self):
        cfg = self.cfg

        vs, pis = self.network(self.rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape))
        dist = Categorical(logits=pis)
        log_probs = dist.log_prob(self.rollouts.actions.view(-1)).unsqueeze(-1)

        advs = self.rollouts.returns[:-1].view(-1, 1) - vs
        value_loss = advs.pow(2).mean()
        policy_loss = (advs.detach() * log_probs).mean().neg()
        entropy = dist.entropy().mean()

        loss = value_loss * cfg.value_loss_coef + policy_loss - cfg.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        self.logger.store(Loss=loss.item())
        self.logger.store(VLoss=value_loss.item())
        self.logger.store(PLoss=policy_loss.item())
        self.logger.store(Entropy=entropy)
        self.logger.store(Loss=loss)


        self.rollouts.obs[0].copy_(self.rollouts.obs[-1])
        self.rollouts.masks[0].copy_(self.rollouts.masks[-1])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self):
        cfg = self.cfg
        logger = self.logger
        logger.store(TrainEpRet=0, VLoss=0, PLoss=0, Entropy=0)
        t0 = time.time()

        states = self.envs.reset()
        self.rollouts.obs[0].copy_(self.state_normalizer(states))
        while self.total_steps < cfg.max_steps:

            self.step()
            self.update()

            if self.total_steps % cfg.log_interval == 0:
                logger.log_tabular('TotalEnvInteracts', self.total_steps)
                logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
                logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
                logger.log_tabular('TrainEpRet', with_min_and_max=True)
                logger.log_tabular('Loss', average_only=True)
                logger.log_tabular('VLoss', average_only=True)
                logger.log_tabular('PLoss', average_only=True)
                logger.log_tabular('Entropy', average_only=True)
                logger.log_tabular('RemHrs', (cfg.max_steps - self.total_steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
                t0 = time.time()
                logger.dump_tabular(self.total_steps)

            if self.total_steps % cfg.save_interval == 0:
                self.save(f'{cfg.ckpt_dir}/{self.total_steps:08d}')
