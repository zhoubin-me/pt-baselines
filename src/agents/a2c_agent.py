import torch
from torch.distributions import Categorical, Normal
from gym.spaces import Box, Discrete
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import time
from collections import namedtuple

from .base_agent import BaseAgent
from src.common.make_env import make_vec_envs
from src.common.model import ConvNet, MLPNet, SepBodyMLP, SepBodyConv
from src.common.logger import EpochLogger
from src.common.normalizer import SignNormalizer, ImageNormalizer
from src.common.utils import tensor

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'action_log_probs', 'rewards', 'values', 'masks', 'badmasks', 'returns', 'gaes'])

class A2CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A2CAgent, self).__init__(cfg)

        self.envs = make_vec_envs(cfg.game, seed=cfg.seed, num_processes=cfg.num_processes, log_dir=cfg.log_dir, allow_early_resets=False, env_type=cfg.env_type)
        self.test_env = make_vec_envs(cfg.game, seed=cfg.seed, num_processes=1, log_dir=cfg.log_dir, allow_early_resets=False, env_type=cfg.env_type, is_test=True)

        if cfg.env_type == 'atari':
            NET = SepBodyConv if cfg.sep_body else ConvNet
            self.network = NET(4, self.envs.action_space.n).cuda()
            self.reward_normalizer = SignNormalizer()
            self.state_normalizer = ImageNormalizer()
            self.action_store_dim = 1
        elif cfg.env_type == 'mujoco' or cfg.env_type == 'bullet':
            NET = SepBodyMLP if cfg.sep_body else MLPNet
            self.network = NET(self.envs.observation_space.shape[0], self.envs.action_space.shape[0]).cuda()
            self.reward_normalizer = lambda x: x
            self.state_normalizer = lambda x: x
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
            scheduler = lambda step : 1 - step * cfg.num_processes * cfg.mini_steps / cfg.max_steps
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler)
        else:
            self.lr_scheduler = None

        self.logger = EpochLogger(cfg.log_dir, exp_name=cfg.algo)

        self.rollouts = Rollouts(
            obs = torch.zeros(cfg.mini_steps + 1, cfg.num_processes,  * self.envs.observation_space.shape).cuda(),
            actions = torch.zeros(cfg.mini_steps, cfg.num_processes, self.action_store_dim).cuda(),
            action_log_probs = torch.zeros(cfg.mini_steps, cfg.num_processes, 1).cuda(),
            values = torch.zeros(cfg.mini_steps + 1, cfg.num_processes, 1).cuda(),
            rewards = torch.zeros(cfg.mini_steps, cfg.num_processes, 1).cuda(),
            masks = torch.zeros(cfg.mini_steps + 1, cfg.num_processes, 1).cuda(),
            badmasks = torch.zeros(cfg.mini_steps + 1, cfg.num_processes, 1).cuda(),
            returns = torch.zeros(cfg.mini_steps + 1, cfg.num_processes, 1).cuda(),
            gaes = torch.zeros(cfg.mini_steps + 1, cfg.num_processes, 1).cuda()
        )

        self.total_steps = 0

    def eval_step(self, states):
        v, pi = self.network(self.state_normalizer(states))
        if isinstance(self.envs.action_space, Discrete):
            dist = Categorical(logits=pi)
            actions = dist.sample()
        elif isinstance(self.envs.action_space, Box):
            dist = Normal(pi, self.network.p_log_std.expand_as(pi).exp())
            actions = dist.sample()
        else:
            raise NotImplementedError('No such action space')
        return actions

    def step(self):
        cfg = self.cfg
        rollouts = self.rollouts
        with torch.no_grad():
            for step in range(cfg.mini_steps):

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
                masks = 1.0 - dones
                badmasks = tensor([[0.0] if 'TimeLimit.truncated' in info else [1.0] for info in infos])

                rollouts.masks[step + 1].copy_(masks)
                rollouts.badmasks[step + 1].copy_(badmasks)
                rollouts.actions[step].copy_(actions)
                rollouts.values[step].copy_(v)
                rollouts.action_log_probs[step].copy_(action_log_probs)
                rollouts.rewards[step].copy_(rewards)
                rollouts.obs[step + 1].copy_(self.state_normalizer(states))



                for info in infos:
                    if 'episode' in info:
                        self.logger.store(TrainEpRet=info['episode']['r'])

            # Compute R and GAE
            v_next, _ = self.network(self.rollouts.obs[-1])

            rollouts.values[-1].copy_(v_next)
            rollouts.returns[-1].copy_(v_next)
            rollouts.gaes[-1].zero_()

            for step in reversed(range(cfg.mini_steps)):
                R = rollouts.returns[step + 1] * cfg.gamma * rollouts.masks[step + 1] + rollouts.rewards[step]
                rollouts.returns[step] = R * rollouts.badmasks[step+ 1] + (1 - rollouts.badmasks[step + 1]) * rollouts.values[step + 1]

                delta = rollouts.rewards[step] + cfg.gamma * rollouts.values[step + 1] * rollouts.masks[step + 1] - rollouts.values[step]
                rollouts.gaes[step] = (delta + cfg.gamma * cfg.gae_lambda * rollouts.masks[step + 1] * rollouts.gaes[step+1]) * rollouts.badmasks[step + 1]


    def sample(self):
        cfg = self.cfg
        rollouts = self.rollouts
        batch_size = cfg.num_processes * cfg.mini_steps
        mini_batch_size = batch_size // cfg.num_mini_batch

        gaes = rollouts.gaes[:-1].view(-1, 1)
        advs = (gaes - gaes.mean()) / (gaes.std() + 1e-5)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = rollouts.obs[:-1].view(-1, *self.envs.observation_space.shape)[indices]
            action_batch = rollouts.actions.view(-1, self.action_store_dim)[indices]
            action_log_prob_batch = rollouts.action_log_probs.view(-1, 1)[indices]
            value_batch = rollouts.values[:-1].view(-1, 1)[indices]
            mask_batch = rollouts.masks[:-1].view(-1, 1)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]
            gae_batch = rollouts.gaes[:-1].view(-1, 1)[indices]
            adv_batch = advs[indices]
            yield obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch

    def update(self):
        cfg = self.cfg
        for epoch in range(cfg.mini_epoches):
            sampler = self.sample()
            for batch_data in sampler:
                obs_batch, action_batch, value_batch, return_batch, mask_batch, action_log_prob_batch, gae_batch, adv_batch = batch_data

                vs, pis = self.network(obs_batch)

                if isinstance(self.envs.action_space, Discrete):
                    dist = Categorical(logits=pis)
                    log_probs = dist.log_prob(action_batch.view(-1)).unsqueeze(-1)
                    entropy = dist.entropy().mean()
                elif isinstance(self.envs.action_space, Box):
                    dist = Normal(pis, self.network.p_log_std.expand_as(pis).exp())
                    log_probs = dist.log_prob(action_batch.view(-1, self.action_store_dim)).sum(-1, keepdim=True)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    raise NotImplementedError('No such action space')

                value_loss = (return_batch + adv_batch - vs).pow(2).mean() * 0.5
                policy_loss = ((return_batch + adv_batch - vs).detach() * log_probs).mean().neg()
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
                test_returns = self.eval_episodes()
                logger.add_scalar('AverageTestEpRet', np.mean(test_returns), self.total_steps)
                test_tabular = {
                    "Epoch": self.total_steps // cfg.save_interval,
                    "Steps": self.total_steps,
                    "NumOfEp": len(test_returns),
                    "AverageTestEpRet": np.mean(test_returns),
                    "StdTestEpRet": np.std(test_returns),
                    "MaxTestEpRet": np.max(test_returns),
                    "MinTestEpRet": np.min(test_returns)}
                logger.dump_test(test_tabular)