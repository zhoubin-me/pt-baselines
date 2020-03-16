import torch.multiprocessing as mp
from collections import deque
import copy
# Copied from ShangtongZhang/DeepRL
# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/BaseAgent.py

class AsyncActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5
    INF = 6

    def __init__(self, cfg):
        mp.Process.__init__(self)
        self.cfg = cfg
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._env = None
        self._network = None
        self._total_steps = 0
        self._cache_len = 2


    def _sample(self):
        transitions = []
        for _ in range(self.cfg.sgd_update_frequency):
            transitions.append(self._transition())
        return transitions

    def run(self):
        self._set_up()
        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            elif op == self.INF:
                while True:
                    self._transition()
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        self.__pipe.send([self.NETWORK, net])


