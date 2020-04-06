import torch
import copy
import time
import torch.multiprocessing as mp
import numpy as np
from collections import deque

from .base_agent import BaseAgent
from .async_actor import AsyncActor
from src.common.async_replay import AsyncReplayBuffer
from src.common.utils import close_obj
from src.common.make_env import make_atari_env
from src.common.schedule import LinearSchedule
from src.common.normalizer import ImageNormalizer, SignNormalizer
from src.common.logger import EpochLogger
from src.common.model import C51Net

class RainbowActor(AsyncActor):
    def __init__(self, cfg, lock, device_id):
        super(RainbowActor, self).__init__(cfg)
        self.lock = lock
        self.device_id = device_id
        self.start()

    def _set_up(self):
        cfg = self.cfg
        self._atoms = torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms).to(self.device_id)
        self._device = torch.device(f'cuda:{self.device_id}') if self.device_id >= 0 else torch.device('cpu')

        self._env = make_atari_env(
            game=cfg.game,
            log_prefix=f'{cfg.log_dir}/train',
            record_video=False,
            seed=cfg.seed
        )()

        self._random_action_prob = LinearSchedule(1.0, cfg.min_epsilon, cfg.epsilon_steps)
        self._state_normalizer = ImageNormalizer()


    def _transition(self):
        if self._state is None:
            self._state = self._env.reset()

        cfg = self.cfg

        if  cfg.noisy or (self._total_steps > cfg.exploration_steps and np.random.rand() > self._random_action_prob()):
            state = torch.from_numpy(self._state_normalizer([self._state])).float().to(self.device)
            with torch.no_grad():
                probs, _ = self._network(state)
            action = (probs * self._atoms).sum(-1).argmax(dim=-1)
            action = action.item()
        else:
            action = self._env.action_space.sample()


        next_state, reward, done, info = self._env.step(action)
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        self._state = next_state
        if done:
            self._state = self._env.reset()
        return entry


class RainbowAgent(BaseAgent):
    def __init__(self, cfg):
        super(RainbowAgent, self).__init__(cfg)
        self.lock = mp.Lock()
        self.device = torch.device(f'cuda:{cfg.device_id}') if cfg.device_id >= 0 else torch.device('cpu')
        self.actor = RainbowActor(cfg, self.lock, cfg.device_id)
        self.test_env = make_atari_env(
            game=cfg.game,
            log_prefix=f'{cfg.log_dir}/test',
            record_video=False,
            episode_life=False,
            seed=cfg.seed
        )()

        self.logger = EpochLogger(cfg.log_dir, exp_name=cfg.algo)
        self.replay = AsyncReplayBuffer(
            buffer_size=cfg.replay_size,
            batch_size=cfg.batch_size,
            prioritize=cfg.prioritize,
            alpha=cfg.replay_alpha,
            beta0=cfg.replay_beta0,
            device_id=cfg.device_id
        )

        self.beta_schedule = LinearSchedule(cfg.replay_beta0, 1.0, cfg.max_steps)

        self.network = C51Net(
            action_dim=self.test_env.action_space.n,
            num_atoms=cfg.num_atoms,
            noisy=cfg.noisy,
            duel=cfg.dueling,
            in_channels=cfg.history_length
        ).to(self.device)

        self.network.train()
        self.network.share_memory()
        self.target_network = copy.deepcopy(self.network)

        if cfg.noisy:
            self.network.train()
            self.target_network.train()

        self.actor.set_network(self.network)

        self.optimizer = torch.optim.Adam(
            params=self.network.parameters(),
            lr=cfg.adam_lr,
            eps=cfg.adam_eps
        )

        self.tracker = deque(maxlen=cfg.nstep)
        self.batch_indices = torch.arange(cfg.batch_size).to(self.device)
        self.atoms = torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms).to(self.device)
        self.delta_atom = (cfg.v_max - cfg.v_min) / (cfg.num_atoms - 1)

        self.state_normalizer = ImageNormalizer()
        self.reward_normalizer = SignNormalizer()
        self.total_steps = 0

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self):
        if np.random.rand() > self.cfg.test_epsilon:
            state = torch.from_numpy(self.state_normalizer([self.test_state])).float().to(self.device)
            prob, _ = self.network(state)
            q = (prob * self.atoms).sum(-1)
            action = np.argmax(q.tolist())
        else:
            action = self.test_env.action_space.sample()
        return action

    def step(self):
        cfg = self.cfg
        if cfg.noisy:
            self.network.reset_noise(cfg.noise_std)
            self.target_network.reset_noise(cfg.noise_std)

        ## Environment Step
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.total_steps += 1
            reward = self.reward_normalizer(reward)
            self.tracker.append([state, action, reward, done])

            R = 0
            for *_, r, d in reversed(self.tracker):
                R += r + cfg.discount * (1 - d) * R

            D = any([x[-1] for x in self.tracker])
            experiences.append(self.tracker[0][:2] + [R, next_state, D])

            if done:
                self.tracker.clear()
                if isinstance(info, dict):
                    if 'episode' in info:
                        self.logger.store(TrainEpRet=info['episode']['r'])

        self.replay.add_batch(experiences)

        ## Upate
        if self.total_steps > cfg.exploration_steps:
            beta = self.beta_schedule(cfg.sgd_update_frequency)
            experiences = self.replay.sample(beta=beta)
            states, actions, rewards, next_states, terminals, *extras = experiences
            states = self.state_normalizer(states)
            next_states = self.state_normalizer(next_states)
            actions = actions.long()


            with torch.no_grad():
                prob_next, _ = self.target_network(next_states)
                if cfg.double:
                    prob_next_online, _ = self.network(next_states)
                    actions_next = prob_next_online.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
                else:
                    actions_next = prob_next.mul(self.atoms).sum(dim=-1).argmax(dim=-1)
                prob_next = prob_next[self.batch_indices, actions_next, :]

                rewards = rewards.float().unsqueeze(-1)
                terminals = terminals.float().unsqueeze(-1)
                atoms_next = rewards + (cfg.discount ** cfg.nstep) * (1 - terminals) * self.atoms.view(1, -1)

                atoms_next.clamp_(cfg.v_min, cfg.v_max)
                b = (atoms_next - cfg.v_min) / self.delta_atom

                l, u = b.floor().long(), b.ceil().long()
                l[(u > 0) * (l == u)] -= 1
                u[(l < (cfg.num_atoms - 1)) * (l == u)] += 1

                target_prob = torch.zeros_like(prob_next)
                offset = torch.linspace(0, ((cfg.batch_size - 1) * cfg.num_atoms), cfg.batch_size)
                offset = offset.unsqueeze(1).expand(cfg.batch_size, cfg.num_atoms).long().to(self.device)

                target_prob.view(-1).index_add_(0, (l + offset).view(-1), (prob_next * (u.float() - b)).view(-1))
                target_prob.view(-1).index_add_(0, (u + offset).view(-1), (prob_next * (b - l.float())).view(-1))

            _, log_prob = self.network(states)
            log_prob = log_prob[self.batch_indices, actions, :]
            error = 0 - target_prob.mul_(log_prob).sum(dim=-1)

            if cfg.prioritize:
                weights, idxes = extras
                idxes = idxes.int().flatten().tolist()
                prioritise = (error+1e-8).flatten().tolist()
                self.replay.update_priorities(idxes, prioritise)
                loss = (error * weights).mean()
            else:
                loss = error.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.store(Loss=loss.item())

        if self.total_steps % cfg.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def run(self):

        logger = self.logger
        cfg = self.cfg

        t0 = time.time()
        logger.store(TrainEpRet=0, Loss=0)

        while self.total_steps < cfg.max_steps:
            self.step()

            if self.total_steps % cfg.log_interval == 0:
                logger.log_tabular('TotalEnvInteracts', self.total_steps)
                logger.log_tabular('Speed', cfg.log_interval / (time.time() - t0))
                logger.log_tabular('NumOfEp', len(logger.epoch_dict['TrainEpRet']))
                logger.log_tabular('TrainEpRet', with_min_and_max=True)
                logger.log_tabular('Loss', average_only=True)
                logger.log_tabular('RemHrs', (cfg.max_steps - self.total_steps) / cfg.log_interval * (time.time() - t0) / 3600.0)
                t0 = time.time()
                logger.dump_tabular(self.total_steps)

            if self.total_steps % cfg.save_interval == 0:
                self.save(f'{cfg.ckpt_dir}/{self.total_steps:08d}')
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

