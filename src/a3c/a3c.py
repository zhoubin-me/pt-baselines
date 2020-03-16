import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import copy
import time
import numpy as np
from collections import deque

from src.common.async_actor import AsyncActor
from src.common.async_replay import AsyncReplayBuffer
from src.common.base_agent import BaseAgent
from src.common.utils import close_obj, tensor, make_a3c_env
from src.common.schedule import LinearSchedule
from src.common.normalizer import SignNormalizer
from src.common.logger import EpochLogger
from src.a3c.model import ACNet

class A3CActor(AsyncActor):
    def __init__(self, cfg, lock, n):
        self.n = n
        self.lock = lock
        super(A3CActor, self).__init__(cfg)
        self.start()

    def _set_up(self):
        cfg = self.cfg
        self._env = make_a3c_env(
            cfg.game,
            f'{cfg.log_dir}/train_{self.n}',
            False,
            seed=cfg.seed+self.n
        )
        self._reward_normalizer = SignNormalizer()
        self._optimizer = None

    def _transition(self):
        if self._state is None:
            self._state = self._env.reset()
            self._hx, self._cx = torch.zeros(1, 256), torch.zeros(1, 256)

        cfg = self.cfg

        transitions = []
        done, R = False, torch.zeros(1, 1)
        for step in range(cfg.steps_per_transit):
            s = torch.from_numpy(self._state).unsqueeze(0)
            with self.lock:
                v, pi, (hx, cx) = self._network((s, (self.hx, self.cx)))
            prob = F.softmax(pi, -1)
            log_prob = F.log_softmax(pi, -1)
            entropy = 0 - (prob * log_prob).sum(-1, keepdim=True)
            action = prob.multinomial(1)
            log_prob = log_prob[:, action]
            next_state, reward, done, _ = self._env.step(action.item())
            self._total_steps += 1


            if done:
                self._state = self._env.reset()
                self._hx, self._cx = torch.zeros(1, 256), torch.zeros(1, 256)
                break
            else:
                self._state = next_state
                self._hx, self._cx = hx, cx
                s = torch.from_numpy(self._state).unsqueeze(0)

                with self.lock, torch.no_grad():
                    R, _, _ = self._network(s, (self._hx, self._cx))

            transitions.append([v, log_prob, self._reward_normalizer(reward), entropy])


        GAE = torch.zeros(1, 1)
        value_loss, policy_loss, v_prev = 0, 0, R
        for idx, v, log_prob, reward, entropy in enumerate(reversed(transitions)):
            R = cfg.discount * R + reward
            advantage = R - v

            value_loss = value_loss + 0.5 * advantage.pow(2)

            dt = reward + cfg.discount * v_prev - v
            GAE = GAE * cfg.discount * cfg.gae_coef + dt

            policy_loss = policy_loss - log_prob * GAE.detach() - cfg.entropy_coef * entropy
            v_prev = v
        loss = policy_loss + cfg.value_loss_coef * value_loss
        return loss



class A3CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A3CAgent, self).__init__(cfg)
        self.lock = mp.Lock()
        self.actors = [A3CActor(cfg, n, self.lock) for n in range(cfg.num_actors)]
        self.test_env = make_a3c_env(
            cfg.game,
            f'{cfg.log_dir}/test',
            True,
            max_episode_steps=cfg.max_episode_steps,
            seed=cfg.seed
        )
        self.logger = EpochLogger(cfg.log_dir)

        self.network = ACNet(
            in_channels=1,
            action_dim=self.test_env.action_space.n,
        )

        self.network.train()
        self.network.share_memory()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            cfg.adam_lr,
        )

        for actor in self.actors:
            actor.set_network(self.network)

        self.total_steps = 0

    def close(self):
        for actor in self.actors:
            close_obj(actor)

    def eval_step(self, state):
        self.state_normalizer.set_read_only()
        if np.random.rand() > self.cfg.test_epsilon:
            state = torch.from_numpy(self.state_normalizer([state])).float().cuda()
            prob, _ = self.network(state)
            q = (prob * self.atoms).sum(-1)
            action = np.argmax(q.tolist())
        else:
            action = self.test_env.action_space.sample()
        self.state_normalizer.unset_read_only()
        return action

    def step(self):

        for actor in self.actors:
            loss = actor.step()
            for l in loss:
                self.optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()




    def run(self):

        logger = self.logger
        cfg = self.cfg

        t0 = time.time()
        logger.store(TrainEpRet=0, Loss=0)

        while self.total_steps < self.cfg.max_steps:
            self.step()

            if self.total_steps % self.cfg.save_interval == 0:
                self.save(f'{self.cfg.log_dir}/{self.total_steps}')

            if self.total_steps % self.cfg.log_interval == 0:
                logger.log_tabular('TotalEnvInteracts', self.total_steps)
                logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
                logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
                logger.log_tabular('TrainEpRet', with_min_and_max=True)
                logger.log_tabular('Loss', average_only=True)
                logger.log_tabular('RemHrs', (cfg.max_steps - self.total_steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
                t0 = time.time()
                logger.dump_tabular(self.total_steps)

            if self.total_steps % cfg.eval_interval == 0:
                test_returns = self.eval_episodes()
                logger.add_scalar('AverageTestEpRet', np.mean(test_returns), self.total_steps)
                test_tabular = {
                    "Epoch": self.total_steps // cfg.eval_interval,
                    "Steps": self.total_steps,
                    "NumOfEp": len(test_returns),
                    "AverageTestEpRet": np.mean(test_returns),
                    "StdTestEpRet": np.std(test_returns),
                    "MaxTestEpRet": np.max(test_returns),
                    "MinTestEpRet": np.min(test_returns)}
                logger.dump_test(test_tabular)

        self.close()

