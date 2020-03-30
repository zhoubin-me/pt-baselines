import torch
from torch.distributions import Categorical, Normal
from gym.spaces import Box, Discrete

import time
from collections import namedtuple

from .base_agent import BaseAgent
from src.common.make_env import make_vec_envs
from src.common.model import ConvNet, MLPNet
from src.common.logger import EpochLogger
from src.common.normalizer import SignNormalizer, ImageNormalizer, MeanStdNormalizer

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'action_log_probs', 'rewards', 'values', 'masks', 'returns'])

class A2CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A2CAgent, self).__init__(cfg)

        self.envs = make_vec_envs(cfg.game, seed=cfg.seed, num_processes=cfg.num_processes, \
                log_dir=cfg.log_dir, allow_early_resets=False, env_type=cfg.env_type)

        if cfg.env_type == 'atari':
            self.network = ConvNet(4, self.envs.action_space.n).cuda()
            self.reward_normalizer = SignNormalizer()
            self.state_normalizer = ImageNormalizer()
            self.action_store_dim = 1
        elif cfg.env_type == 'mujoco':
            self.network = MLPNet(self.envs.observation_space.shape[0], self.envs.action_space.shape[0]).cuda()
            self.reward_normalizer = MeanStdNormalizer(clip=10)
            self.state_normalizer = MeanStdNormalizer(clip=5)
            self.action_store_dim = self.envs.action_space.shape[0]
        else:
            raise NotImplementedError("No such environment")

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

        self.rollouts = Rollouts(
            obs = torch.zeros(cfg.nsteps + 1, cfg.num_processes,  * self.envs.observation_space.shape).cuda(),
            actions = torch.zeros(cfg.nsteps, cfg.num_processes, self.action_store_dim).cuda(),
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

                if self.total_steps == 0:
                    states = self.envs.reset()
                    self.rollouts.obs[0].copy_(self.state_normalizer(states))
                elif step == 0:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.rollouts.obs[0].copy_(self.rollouts.obs[-1])
                    self.rollouts.masks[0].copy_(self.rollouts.masks[-1])

                v, pi = self.network(self.rollouts.obs[step])

                if isinstance(self.envs.action_space, Discrete):
                    dist = Categorical(logits=pi)
                    actions = dist.sample()
                    action_log_probs = dist.log_prob(actions)
                    actions = actions.unsqueeze(-1)
                    action_log_probs = action_log_probs.unsqueeze(-1)

                elif isinstance(self.envs.action_space, Box):
                    dist = Normal(pi, self.network.p_log_std.expand_as(pi).exp())
                    actions = dist.sample()
                    action_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
                else:
                    raise NotImplementedError('No such action space')

                states, rewards, dones, infos = self.envs.step(actions)
                self.total_steps += cfg.num_processes
                rewards = self.reward_normalizer(rewards)

                self.rollouts.masks[step + 1].copy_(1 - dones)
                self.rollouts.actions[step].copy_(actions)
                self.rollouts.values[step].copy_(v)
                self.rollouts.action_log_probs[step].copy_(action_log_probs)
                self.rollouts.rewards[step].copy_(rewards)
                self.rollouts.obs[step + 1].copy_(self.state_normalizer(states))

                for info in infos:
                    if 'episode' in info:
                        self.logger.store(TrainEpRet=info['episode']['r'])

            # Compute R and GAE
            v_next, _ = self.network(self.rollouts.obs[-1])

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

        kwargs = {
            'Loss': loss.item(),
            'VLoss': value_loss.item(),
            'PLoss': policy_loss.item(),
            'Entropy': entropy.item(),
        }

        self.logger.store(**kwargs)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def run(self):
        cfg = self.cfg
        logger = self.logger
        logger.store(TrainEpRet=0, Loss=0, VLoss=0, PLoss=0, Entropy=0)
        t0 = time.time()
        last_epoch = -1

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

            epoch = self.total_steps // self.cfg.save_interval
            if epoch > last_epoch:
                self.save(f'{cfg.ckpt_dir}/{self.total_steps:08d}')
                last_epoch = epoch
