import torch
import torch.multiprocessing as mp

from src.common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.common.utils import tensor

# Copied from ShangtongZhang/DeepRL
# https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/replay.py

class AsyncReplayBuffer(mp.Process):
    FEED = 0
    SAMPLE = 1
    EXIT = 2
    FEED_BATCH = 3
    UPDATE = 4

    def __init__(self, buffer_size, batch_size, prioritize=False, alpha=0.5, beta0=0.4, device_id=-1):
        mp.Process.__init__(self)
        self.pipe, self.worker_pipe = mp.Pipe()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.prioritize = prioritize
        self.alpha = alpha
        self.beta0 = beta0
        self.cache_len = 2
        self.device_id = device_id
        self.start()

    def run(self):
        if self.prioritize:
            replay = PrioritizedReplayBuffer(self.buffer_size, self.alpha)
        else:
            replay = ReplayBuffer(self.buffer_size)

        cache = []
        pending_batch = None
        device = torch.device('cpu') if self.device_id < 0 else torch.device(f'cuda:{self.device_id}')

        first = True
        cur_cache = 0

        def set_up_cache():
            batch_data = replay.sample(self.batch_size, beta=self.beta0)
            batch_data = [torch.tensor(x).to(device) for x in batch_data]
            for i in range(self.cache_len):
                cache.append([x.clone() for x in batch_data])
                for x in cache[i]: x.share_memory_()
            sample(0, self.beta0)
            sample(1, self.beta0)

        def sample(cur_cache, beta):
            batch_data = replay.sample(batch_size=self.batch_size, beta=beta)
            batch_data = [torch.tensor(x, device=device) for x in batch_data]
            for cache_x, x in zip(cache[cur_cache], batch_data):
                cache_x.copy_(x)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.FEED:
                replay.add(*data)
            elif op == self.FEED_BATCH:
                if not first:
                    pending_batch = data
                else:
                    for transition in data:
                        replay.add(*transition)
            elif op == self.SAMPLE:
                if first:
                    set_up_cache()
                    first = False
                    self.worker_pipe.send([cur_cache, cache])
                else:
                    self.worker_pipe.send([cur_cache, None])
                cur_cache = (cur_cache + 1) % 2
                sample(cur_cache, beta=data)
                if pending_batch is not None:
                    for transition in pending_batch:
                        replay.add(*transition)
                    pending_batch = None
            elif op == self.UPDATE:
                replay.update_priorities(*data)
            elif op == self.EXIT:
                self.worker_pipe.close()
                return
            else:
                raise Exception('Unknown command')

    def add(self, exp):
        self.pipe.send([self.FEED, exp])

    def add_batch(self, exps):
        self.pipe.send([self.FEED_BATCH, exps])

    def update_priorities(self, idxs, priorities):
        self.pipe.send([self.UPDATE, (idxs, priorities)])

    def sample(self, beta=None):
        self.pipe.send([self.SAMPLE, beta])
        cache_id, data = self.pipe.recv()
        if data is not None:
            self.cache = data
        return self.cache[cache_id]

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()

