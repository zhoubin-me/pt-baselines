import os
import torch
from delegans.common.bench import _atari7


print(_atari7)

N = torch.cuda.device_count()
for i, game in enumerate(_atari7):
    os.system(f'export CUDA_VISIBLE_DEVICES="{(i + 2)% N}";python -m delegans.run --game {game} &')
