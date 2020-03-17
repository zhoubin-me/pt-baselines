

## Asynchronous Advantage Actor Critic (A3C) RL

Our implementation is based on 

https://github.com/ikostrikov/pytorch-a3c

We notice they use different weights initialization strategy.

Without using their strategy, it needs 30min (2M) to converge for Pong.

With their strategy, it only needs 10min (<1M) to converge.