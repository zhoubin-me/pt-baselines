import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import time
import numpy as np

from collections import deque

from src.common.base_agent import BaseAgent
from src.common.utils import close_obj, tensor, make_a3c_env
from src.common.normalizer import SignNormalizer
from src.common.logger import EpochLogger
from .model import ACNet



class A3CActor(mp.Process):

    NETWORK = 4
    def __init__(self, cfg, n, lock, counter):
        super(A3CActor, self).__init__()
        self.n = n
        self.lock = lock
        self.counter = counter
        self.cfg = cfg
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._env = None
        self._network = None
        self._total_steps = 0

    def _set_up(self):
        cfg = self.cfg
        self._env = make_a3c_env(
            cfg.game,
            f'{cfg.log_dir}/train_{self.n}',
            False,
            max_episode_steps = cfg.max_episode_steps,
            seed=cfg.seed+self.n
        )()

        self._reward_normalizer = SignNormalizer()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])


    def run(self):
        cfg = self.cfg
        self._set_up()

        # Setup Netowrk
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.NETWORK:
                self._network = data
                self._optimizer = torch.optim.Adam(self._network.parameters(), cfg.adam_lr)
                break


        done, rs = True, None
        while True:
            rollouts = deque()
            if done:
                state, hx, cx = self._env.reset(), torch.zeros(1, 256), torch.zeros(1, 256)
            else:
                hx, cx = hx.detach(), cx.detach()

            # Sample experiences
            for step in range(cfg.nsteps):
                v, pi, (hx, cx) = self._network((torch.from_numpy(state).unsqueeze(0),(hx, cx)))
                m = Categorical(logits=pi)

                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = m.entropy()

                state, reward, done, info = self._env.step(action.item())
                reward = self._reward_normalizer(reward)

                with self.lock:
                    self.counter.value += 1

                rollouts.append((v.view(-1), log_prob.view(-1), entropy.view(-1), reward))

                if done:
                    if 'episode' in info:
                        rs = info['episode']['r']
                    else:
                        rs = None
                    break
                else:
                    rs = None


            # Update network
            v, _, _ = self._network((torch.from_numpy(state).unsqueeze(0), (hx, cx)))
            R = v.view(-1).detach() if not done else 0
            policy_loss, value_loss, gae, v_next = 0, 0, 0, R
            for v, log_prob, entropy, reward in reversed(rollouts):
                R = cfg.discount * R + reward
                advantage = R - v

                td_error = reward + cfg.discount * v_next - v
                gae = gae * cfg.discount * cfg.gae_coef + td_error

                value_loss += 0.5 * advantage * advantage
                policy_loss += 0 - log_prob * gae.detach() - cfg.entropy_coef * entropy
                v_next = v


            self._optimizer.zero_grad()
            loss = policy_loss + cfg.value_loss_coef * value_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._network.parameters(), cfg.max_grad_norm)
            with self.lock:
                self._optimizer.step()
            self._optimizer.step()

            if done:
                print(f"rank {self.n:2d},\t loss {loss.item():6.3f},\t rt {rs:5.0f},")




class A3CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A3CAgent, self).__init__(cfg)
        self.lock = mp.Lock()
        self.counter = mp.Value('i', 0)

        self.actors = [A3CActor(cfg, n, self.lock, self.counter) for n in range(cfg.num_actors)]
        self.test_env = make_a3c_env(
            cfg.game,
            f'{cfg.log_dir}/test',
            True,
            max_episode_steps=cfg.max_episode_steps,
            seed=cfg.seed
        )()
        self.logger = EpochLogger(cfg.log_dir)

        self.network = ACNet(self.test_env.observation_space.shape[0], self.test_env.action_space.n)

        self.network.train()
        self.network.share_memory()

    def close(self):
        for actor in self.actors:
            close_obj(actor)

    def eval_step(self, state):
        s = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            _, pi, (self.hx, self.cx) = self.network((s, (self.hx, self.cx)))
        action = pi.argmax(dim=-1)
        return action.item()

    def eval_episode(self):
        env = self.test_env
        state = env.reset()
        self.hx, self.cx = torch.zeros(1, 256), torch.zeros(1, 256)
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            if isinstance(info, dict):
                if 'episode' in info:
                    ret = info['episode']['r']
                    break
        return ret

    def run(self):

        for actor in self.actors:
            actor.start()
            actor.set_network(self.network)


        logger = self.logger
        t0 = time.time()

        while self.counter.value < self.cfg.max_steps:
            test_returns = self.eval_episodes()
            logger.add_scalar('AverageTestEpRet', np.mean(test_returns), self.counter.value)
            test_tabular = {
                "Steps": self.counter.value,
                "Speed": self.counter.value / (time.time() - t0),
                "NumOfEp": len(test_returns),
                "AverageTestEpRet": np.mean(test_returns),
                "StdTestEpRet": np.std(test_returns),
                "MaxTestEpRet": np.max(test_returns),
                "MinTestEpRet": np.min(test_returns),
                "WallTime(min)": (time.time() - t0) / 60}
            logger.dump_test(test_tabular)

        for actor in self.actors:
            actor.join()

        self.close()

