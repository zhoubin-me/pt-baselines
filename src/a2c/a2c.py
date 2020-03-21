import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import time
import numpy as np
from collections import deque, namedtuple

from src.common.base_agent import BaseAgent
from .env import make_vec_envs
from .model import ACNet
from src.common.logger import EpochLogger
from src.common.normalizer import SignNormalizer

Rollouts = namedtuple('Rollouts', ['obs', 'actions', 'rewards', 'values', 'masks', 'returns'])

class A2CAgent(BaseAgent):
    def __init__(self, args):
        super(A2CAgent, self).__init__(args)

        self.envs = make_vec_envs(args.game, args.seed, args.num_processes,
                      args.gamma, args.log_dir, torch.device(args.device_id), False)

        self.network = ACNet(4, self.envs.action_space.n).cuda()
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), args.rms_lr, eps=args.rms_eps, alpha=args.rms_alpha)
        self.logger = EpochLogger(args.log_dir)
        self.reward_normalizer = SignNormalizer()



        self.rollouts = Rollouts(
            obs = torch.zeros(args.nsteps + 1, args.num_processes,  * self.envs.observation_space.shape).cuda(),
            actions = torch.zeros(args.nsteps, args.num_processes, 1).cuda(),
            values = torch.zeros(args.nsteps + 1, args.num_processes, 1).cuda(),
            rewards = torch.zeros(args.nsteps, args.num_processes, 1).cuda(),
            masks = torch.zeros(args.nsteps + 1, args.num_processes, 1).cuda(),
            returns = torch.zeros(args.nsteps + 1, args.num_processes, 1).cuda()
        )


    def run(self):

        cfg = self.cfg
        logger = self.logger
        logger.store(TrainEpRet=0, Loss=0)
        t0 = time.time()

        states = self.envs.reset()
        self.rollouts.obs.copy_(states / 255.0)
        steps = 0
        while steps < cfg.max_steps:
            # Sample experiences
            with torch.no_grad():
                for step in range(cfg.nsteps):
                    v, pi = self.network(self.rollouts.obs[step])
                    actions = Categorical(logits=pi).sample()

                    states, rewards, dones, infos = self.envs.step(actions)
                    # rewards = self.reward_normalizer(rewards)
                    steps += cfg.num_processes

                    self.rollouts.masks[step + 1].copy_(torch.tensor(1 - dones).unsqueeze(-1).float().cuda())
                    self.rollouts.actions[step].copy_(torch.tensor(actions).unsqueeze(-1).long().cuda())
                    self.rollouts.values[step].copy_(v)
                    self.rollouts.rewards[step].copy_(torch.tensor(rewards).float().cuda())
                    self.rollouts.obs[step + 1].copy_(states / 255.0)


                    for info in infos:
                        if 'episode' in info:
                            self.logger.store(TrainEpRet=info['episode']['r'])

                # Compute R and GAE
                v_next, _, = self.network(self.rollouts.obs[-1])
                self.rollouts.values[-1].copy_(v_next)
                gae = 0
                for step in reversed(range(cfg.nsteps)):
                    delta = self.rollouts.rewards[step] + cfg.gamma * self.rollouts.values[step + 1] * self.rollouts.masks[step + 1] - self.rollouts.values[step]
                    gae = delta + cfg.gamma * cfg.gae_lambda * self.rollouts.masks[step + 1] * gae
                    self.rollouts.returns[step].copy_(gae + self.rollouts.values[step])


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
            logger.store(Loss=loss.item())


            self.rollouts.obs[0].copy_(self.rollouts.obs[-1])
            self.rollouts.masks[0].copy_(self.rollouts.masks[-1])


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




            # Update networks
            # sx = torch.cat([x[0] for x in rollouts], 0)
            # ax = torch.cat([x[1] for x in rollouts], 0).long()
            #
            # rs = torch.cat(list(reversed(Rs)), 0)
            # gaes = torch.cat(list(reversed(GAEs)), 0)
            # vs, pis = self.network(sx)
            #
            # dist = Categorical(logits=pis)
            # log_probs = dist.log_prob(ax)
            # value_loss = (rs - vs).pow(2)
            # policy_loss = 0 - gaes * log_probs - cfg.entropy_coef * dist.entropy()
            #
            # loss = policy_loss.mean() + cfg.value_loss_coef * value_loss.mean()
            #
            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
            # self.optimizer.step()
            # logger.store(Loss=loss.item())
            #
            # if steps % cfg.log_interval == 0:
            #     logger.log_tabular('TotalEnvInteracts', steps)
            #     logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
            #     logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
            #     logger.log_tabular('TrainEpRet', with_min_and_max=True)
            #     logger.log_tabular('Loss', average_only=True)
            #     logger.log_tabular('RemHrs',
            #                        (cfg.max_steps - steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
            #     t0 = time.time()
            #     logger.dump_tabular(steps)

        # for j in range(num_updates):
        #
        #     for step in range(args.nsteps):
        #         # Sample actions
        #         with torch.no_grad():
        #             value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
        #                 rollouts.obs[step], rollouts.recurrent_hidden_states[step],
        #                 rollouts.masks[step])
        #
        #         # Obser reward and next obs
        #         obs, reward, done, infos = envs.step(action)
        #
        #         for info in infos:
        #             if 'episode' in info.keys():
        #                 episode_rewards.append(info['episode']['r'])
        #
        #         # If done then clean the history of observations.
        #         masks = torch.FloatTensor(
        #             [[0.0] if done_ else [1.0] for done_ in done])
        #         bad_masks = torch.FloatTensor(
        #             [[0.0] if 'bad_transition' in info.keys() else [1.0]
        #              for info in infos])
        #         rollouts.insert(obs, recurrent_hidden_states, action,
        #                         action_log_prob, value, reward, masks, bad_masks)
        #
        #     with torch.no_grad():
        #         next_value = self.policy.get_value(
        #             rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
        #             rollouts.masks[-1]).detach()
        #
        #
        #     rollouts.compute_returns(next_value, True, args.gamma,
        #                              args.gae_lambda, False)
        #
        #     obs_shape = rollouts.obs.size()[2:]
        #     action_shape = rollouts.actions.size()[-1]
        #     num_steps, num_processes, _ = rollouts.rewards.size()
        #
        #     values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
        #         rollouts.obs[:-1].view(-1, *obs_shape),
        #         rollouts.recurrent_hidden_states[0].view(
        #             -1, 512),
        #         rollouts.masks[:-1].view(-1, 1),
        #         rollouts.actions.view(-1, action_shape))
        #
        #     values = values.view(num_steps, num_processes, 1)
        #     action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        #
        #     advantages = rollouts.returns[:-1] - values
        #     value_loss = advantages.pow(2).mean()
        #
        #     action_loss = -(advantages.detach() * action_log_probs).mean()
        #
        #
        #     self.optimizer.zero_grad()
        #     (value_loss * args.value_loss_coef + action_loss -
        #      dist_entropy * args.entropy_coef).backward()
        #
        #     torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        #
        #     self.optimizer.step()
        #
        #     rollouts.after_update()
        #
        #
        #     if j % 10 == 0 and len(episode_rewards) > 1:
        #         total_num_steps = (j + 1) * args.num_processes * args.nsteps
        #         end = time.time()
        #         print(
        #             "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
        #                 .format(j, total_num_steps,
        #                         int(total_num_steps / (end - start)),
        #                         len(episode_rewards), np.mean(episode_rewards),
        #                         np.median(episode_rewards), np.min(episode_rewards),
        #                         np.max(episode_rewards), dist_entropy, value_loss,
        #                         action_loss))






