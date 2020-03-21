import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import time
import numpy as np
from collections import deque

from src.common.base_agent import BaseAgent
from .env import make_vec_envs
from .model import Policy
from .storage import RolloutStorage

class A2CAgent(BaseAgent):
    def __init__(self, args):
        super(A2CAgent, self).__init__(args)

        self.envs = make_vec_envs(f'{args.game}NoFrameskip-v4', args.seed, args.num_processes,
                      args.gamma, args.log_dir, torch.device(args.device_id), False)

        self.policy = Policy(self.envs.observation_space.shape, self.envs.action_space.n, base_kwargs={'recurrent': True}).cuda()
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), args.rms_lr, eps=args.rms_eps, alpha=args.rms_alpha)


        self.rollouts = RolloutStorage(
            args.nsteps, args.num_actors, self.envs.observation_space.shape, self.envs.action_space, 512
        )

    def run(self):

        args = self.cfg
        envs = self.envs
        rollouts = self.rollouts
        device = torch.device(args.device_id)
        obs = self.envs.reset()
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(torch.device(args.device_id))

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(
            args.max_steps) // args.nsteps // args.num_actors
        for j in range(num_updates):

            for step in range(args.nsteps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.policy.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, True, args.discount,
                                     args.gae_coef, False)

            obs_shape = rollouts.obs.size()[2:]
            action_shape = rollouts.actions.size()[-1]
            num_steps, num_processes, _ = rollouts.rewards.size()

            values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(
                    -1, 512),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, action_shape))

            values = values.view(num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()


            self.optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss -
             dist_entropy * args.entropy_coef).backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)

            self.optimizer.step()

            rollouts.after_update()


            if j % 10 == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_actors * args.nsteps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards), dist_entropy, value_loss,
                                action_loss))






