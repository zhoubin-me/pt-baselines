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
            seed=cfg.seed+self.n
        )
        self._reward_normalizer = SignNormalizer()
        self._network = ACNet(self._env.observation_space.shape[0], self._env.action_space)
        self._network.train()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])

    def run(self):
        cfg = self.cfg
        self._set_up()

        state = self._env.reset()
        state = torch.from_numpy(state)
        done = True
        episode_length = 0
        rs = 0

        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.NETWORK:
                self._shared_net = data
                self._optimizer = torch.optim.Adam(self._shared_net.parameters(), self.cfg.adam_lr)
                break

        while True:
            # Sync with the shared model
            self._network.load_state_dict(self._shared_net.state_dict())
            if done:
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
            else:
                cx = cx.detach()
                hx = hx.detach()

            values = []
            log_probs = []
            rewards = []
            entropies = []

            for step in range(cfg.steps_per_transit):
                episode_length += 1
                value, logit, (hx, cx) = self._network((state.unsqueeze(0),(hx, cx)))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                state, reward, done, _ = self._env.step(action.numpy())
                rs += reward
                reward = self._reward_normalizer(reward)

                with self.lock:
                    self.counter.value += 1

                if done:
                    episode_length = 0
                    state = self._env.reset()

                state = torch.from_numpy(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = self._network((state.unsqueeze(0), (hx, cx)))
                R = value.detach()

            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = cfg.discount * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = rewards[i] + cfg.discount * \
                          values[i + 1] - values[i]
                gae = gae * cfg.discount * cfg.gae_coef + delta_t

                policy_loss = policy_loss - log_probs[i] * gae.detach() - cfg.entropy_coef * entropies[i]

            self._optimizer.zero_grad()

            loss = policy_loss + cfg.value_loss_coef * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), cfg.max_grad_norm)
            ensure_shared_grads(self._network, self._shared_net)
            self._optimizer.step()

            if done:
                print(f"rank {self.n:2d},\t loss {loss.item():6.3f},\t rt {rs:5.0f},")
                rs = 0



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, cfg, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(cfg.seed + rank)

    env = make_a3c_env(
        cfg.game,
        f'{cfg.log_dir}/train',
        True,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed
    )
    env.seed(cfg.seed + rank)

    model = ACNet(env.observation_space.shape[0], env.action_space.n)

    if optimizer is None:
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=cfg.adam_lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(cfg.steps_per_transit):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= cfg.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = cfg.discount * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + cfg.discount * \
                values[i + 1] - values[i]
            gae = gae * cfg.discount * cfg.gae_coef + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - cfg.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + cfg.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()


def test(rank, cfg, shared_model, counter):

    env = make_a3c_env(
        cfg.game,
        f'{cfg.log_dir}/test',
        True,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed
    )

    model = ACNet(env.observation_space.shape[0], env.action_space.n)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= cfg.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)


class A3CAgent(BaseAgent):
    def __init__(self, cfg):
        super(A3CAgent, self).__init__(cfg)


    def run(self):
        processes = []

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        env = make_a3c_env(
            self.cfg.game,
            f'{self.cfg.log_dir}/main',
            True,
            max_episode_steps=self.cfg.max_episode_steps,
            seed=self.cfg.seed
        )

        shared_model = ACNet(
            env.observation_space.shape[0], env.action_space.n)
        shared_model.share_memory()
        optimizer = None

        p = mp.Process(target=test, args=(self.cfg.num_actors, self.cfg, shared_model, counter))
        p.start()
        processes.append(p)

        for rank in range(0, self.cfg.num_actors):
            p = mp.Process(target=train, args=(rank, self.cfg, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # self.lock = mp.Lock()
        # self.counter = mp.Value('i', 0)

        # self.actors = [A3CActor(cfg, n, self.lock, self.counter) for n in range(cfg.num_actors)]
        # self.test_env = make_a3c_env(
        #     cfg.game,
        #     f'{cfg.log_dir}/test',
        #     True,
        #     max_episode_steps=cfg.max_episode_steps,
        #     seed=cfg.seed
        # )
        # self.logger = EpochLogger(cfg.log_dir)

        # self.network = ACNet(
        #     1,
        #     self.test_env.action_space.n,
        # )
        #
        # self.network.train()
        # self.network.share_memory()


    # def close(self):
    #     for actor in self.actors:
    #         close_obj(actor)
    #
    # def eval_step(self, state):
    #     s = torch.from_numpy(state).unsqueeze(0)
    #     with torch.no_grad():
    #         _, pi, (self.hx, self.cx) = self.network((s, (self.hx, self.cx)))
    #     action = pi.argmax(dim=-1)
    #     return action.item()
    #
    # def eval_episode(self):
    #     env = self.test_env
    #     state = env.reset()
    #     self.hx, self.cx = torch.zeros(1, 256), torch.zeros(1, 256)
    #     while True:
    #         action = self.eval_step(state)
    #         state, reward, done, info = env.step(action)
    #         if isinstance(info, dict):
    #             if 'episode' in info:
    #                 ret = info['episode']['r']
    #                 break
    #     return ret

    #   def run(self):

        # for actor in self.actors:
        #     actor.start()
        #
        # for actor in self.actors:
        #     actor.set_network(self.network)
        #
        # logger = self.logger
        # t0 = time.time()
        #
        # while True:
        #     test_returns = self.eval_episodes()
        #     logger.add_scalar('AverageTestEpRet', np.mean(test_returns), self.counter.value)
        #     test_tabular = {
        #         "Steps": self.counter.value,
        #         "Speed": self.counter.value / (time.time() - t0),
        #         "NumOfEp": len(test_returns),
        #         "AverageTestEpRet": np.mean(test_returns),
        #         "StdTestEpRet": np.std(test_returns),
        #         "MaxTestEpRet": np.max(test_returns),
        #         "MinTestEpRet": np.min(test_returns),
        #         "WallTime(min)": (time.time() - t0) / 60}
        #     logger.dump_test(test_tabular)
        #
        # for actor in self.actors:
        #     actor.join()

        #
        # while True:
        #     test_returns = self.eval_episodes()
        #     logger.add_scalar('AverageTestEpRet', np.mean(test_returns), self.total_steps)
        #     test_tabular = {
        #         "Epoch": self.total_steps // cfg.eval_interval,
        #         "Steps": self.total_steps,
        #         "NumOfEp": len(test_returns),
        #         "AverageTestEpRet": np.mean(test_returns),
        #         "StdTestEpRet": np.std(test_returns),
        #         "MaxTestEpRet": np.max(test_returns),
        #         "MinTestEpRet": np.min(test_returns)}
        #     logger.dump_test(test_tabular)
        # N = 0
        # while self.total_steps < self.cfg.max_steps:
        #     for _ in range(cfg.log_interval // (cfg.num_actors * cfg.steps_per_transit)):
        #         self.step()
        #         N += 1
        #
        #     logger.log_tabular('TotalEnvInteracts', self.total_steps)
        #     logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
        #     logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
        #     logger.log_tabular('TrainEpRet', with_min_and_max=True)
        #     logger.log_tabular('Loss', average_only=True)
        #     logger.log_tabular('RemHrs', (cfg.max_steps - self.total_steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
        #     t0 = time.time()
        #     logger.dump_tabular(self.total_steps)
        #
        #     if N % (cfg.save_interval // cfg.log_interval) == 0:
        #         self.save(f'{self.cfg.log_dir}/{self.total_steps}')
        #
        #     if N % (cfg.eval_interval // cfg.log_interval) == 0:

        # self.close()

