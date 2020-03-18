import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import time

from collections import deque

from src.common.base_agent import BaseAgent
from src.common.utils import make_a3c_env, make_vec_env
from src.common.normalizer import SignNormalizer
from src.common.logger import EpochLogger
from src.a2c.model import ACNet



class A2CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A2CAgent, self).__init__(cfg)

        self.envs = make_vec_env(
            cfg.game,
            f'{cfg.log_dir}/train',
            False,
            max_episode_steps=cfg.max_episode_steps,
            seed=cfg.seed,
            gamma=cfg.discount,
            num_processes=cfg.num_envs
        )
        self.logger = EpochLogger(cfg.log_dir)

        self.network = ACNet(4, self.envs.action_space.n).cuda()
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), cfg.rms_lr, eps=cfg.rms_eps, alpha=cfg.rms_alpha)
        self.reward_normalizer = SignNormalizer()

    def run(self):

        cfg = self.cfg
        torch.autograd.set_detect_anomaly(True)
        logger = self.logger
        logger.store(TrainEpRet=0, Loss=0)
        t0 = time.time()

        states, hx, cx = self.envs.reset(), torch.zeros(cfg.num_envs, 512).cuda(), torch.zeros(cfg.num_envs, 512).cuda()
        steps = 0
        while steps < cfg.max_steps:
            # Sample experiences
            rollouts, Rs, GAEs = [], [], []
            with torch.no_grad():
                for step in range(cfg.nsteps):
                    v, pi, (hx_, cx_) = self.network((states, (hx, cx)))
                    actions = Categorical(logits=pi).sample()

                    states, rewards, dones, infos = self.envs.step(actions)
                    rewards = self.reward_normalizer(rewards)
                    steps += cfg.num_envs
                    rollouts.append((states, actions, v, hx, cx, rewards, dones))
                    hx = hx_ * (1 - dones)
                    cx = cx_ * (1 - dones)

                    for info in infos:
                        if 'episode' in info:
                            self.logger.store(TrainEpRet=info['episode']['r'])
                            # print(info)

                # Compute R and GAE
                v_next, _, _ = self.network((states, (hx, cx)))
                policy_loss, value_loss, gae, R = 0, 0, 0, v_next
                for _, _, v, _, _, reward, done in reversed(rollouts):
                    R = cfg.discount * R * (1 - done) + reward

                    td_error = reward + cfg.discount * v_next - v
                    gae = gae * cfg.discount * cfg.gae_coef + td_error

                    v_next = v

                    Rs.append(R)
                    GAEs.append(gae)


            # Update networks
            sx = torch.cat([x[0] for x in rollouts], 0)
            ax = torch.cat([x[1] for x in rollouts], 0).long()
            hxs = torch.cat([x[3] for x in rollouts], 0)
            cxs = torch.cat([x[4] for x in rollouts], 0)

            rs = torch.cat(list(reversed(Rs)), 0)
            gaes = torch.cat(list(reversed(GAEs)), 0)
            vs, pis, _ = self.network((sx, (hxs, cxs)))


            dist = Categorical(logits=pis)
            log_probs = dist.log_prob(ax)
            value_loss = (rs - vs).pow(2)
            policy_loss = 0 - gaes * log_probs - dist.entropy()

            loss = policy_loss.mean() + cfg.value_loss_coef * value_loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            logger.store(Loss=loss.item())


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



