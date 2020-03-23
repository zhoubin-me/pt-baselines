import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import time
import numpy as np
from collections import deque, namedtuple

from src.common.base_agent import BaseAgent
from src.common.utils import make_vec_envs
from src.common.logger import EpochLogger
from src.common.normalizer import SignNormalizer, ImageNormalizer
from src.common.schedule import LinearSchedule
from .model import ACNet

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'rewards', 'values', 'masks', 'returns'])

class PPOAgent(BaseAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

        self.envs = make_vec_envs(cfg.game,
                                  cfg.log_dir,
                                  record_video=False,
                                  seed=cfg.seed,
                                  num_processes=cfg.num_processes,
                                  gamma=cfg.gamma)

        self.network = ACNet(4, self.envs.action_space.n).cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(), cfg.rms_lr, eps=cfg.rms_eps)
        self.lr_schedule = LinearSchedule(cfg.rms_lr, 0, cfg.max_steps // cfg.num_processes)
        self.optimizer_schedule = torch.optim.lr_scheduler.LambdaLR(self.optimizer)
        self.logger = EpochLogger(cfg.log_dir)
        self.reward_normalizer = SignNormalizer()
        self.state_normalizer = ImageNormalizer()
        self.total_steps = 0


        self.rollouts = Rollouts(
            obs = torch.zeros(cfg.nsteps + 1, cfg.num_processes,  * self.envs.observation_space.shape).cuda(),
            actions = torch.zeros(cfg.nsteps, cfg.num_processes, 1).cuda(),
            values = torch.zeros(cfg.nsteps + 1, cfg.num_processes, 1).cuda(),
            rewards = torch.zeros(cfg.nsteps, cfg.num_processes, 1).cuda(),
            masks = torch.zeros(cfg.nsteps + 1, cfg.num_processes, 1).cuda(),
            returns = torch.zeros(cfg.nsteps + 1, cfg.num_processes, 1).cuda()
        )

    def step(self):
        cfg = self.cfg

        if cfg.lr_decrease_schedule:
            self.lr_schedule.step()
        # Sample experiences
        with torch.no_grad():
            for step in range(cfg.nsteps):
                v, pi = self.network(self.rollouts.obs[step])
                actions = Categorical(logits=pi).sample()

                states, rewards, dones, infos = self.envs.step(actions)
                self.total_steps += cfg.num_processes

                self.rollouts.masks[step + 1].copy_(torch.tensor(1 - dones).float().cuda())
                self.rollouts.actions[step].copy_(torch.tensor(actions).unsqueeze(-1).long().cuda())
                self.rollouts.values[step].copy_(v)
                self.rollouts.rewards[step].copy_(torch.tensor(rewards).float().cuda())
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


    def run(self):

        cfg = self.cfg
        logger = self.logger
        logger.store(TrainEpRet=0, Loss=0)
        t0 = time.time()

        states = self.envs.reset()
        self.rollouts.obs.copy_(self.state_normalizer(states))
        steps = 0

        while steps < cfg.max_steps:
            self.step()
            if steps % cfg.log_interval == 0:
                logger.log_tabular('TotalEnvInteracts', steps)
                logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
                logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
                logger.log_tabular('TrainEpRet', with_min_and_max=True)
                logger.log_tabular('Loss', average_only=True)
                logger.log_tabular('RemHrs',
                                   (cfg.max_steps - steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
                t0 = time.time()
                logger.dump_tabular(steps)





