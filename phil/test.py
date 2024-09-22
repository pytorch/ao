# Simplest test cases for comparison

import torch
from torch import tensor, Tensor

# A few steps of standard AdamW FP32

a = tensor([0.8477, 0.3092, 0.2363, 0.2300], device='cuda')        # can use torch.rand(4)
a.grad = tensor([0.8530, 0.7153, 0.1018, 0.4003], device='cuda')   # fake gradient via torch.rand(4)
o = torch.optim.AdamW([a], fused=True)                             # AdamW Optimizer

for i in range(3):
    o.step()
    print("Step " + str(i) + ": " + str(a))

'''
Step 0: a == tensor([0.8467, 0.3082, 0.2353, 0.2290])
Step 1: a == tensor([0.8457, 0.3072, 0.2343, 0.2280])
Step 2: a == tensor([0.8447, 0.3062, 0.2333, 0.2270])

              ^^^ Notice these are different lol ^^^
'''

# Up next: match TorchAO's FP8 quantization

# `python setup.py develop` is giving me errors (ssh-ing Lambdalabs)
# so just copy-pasting instead of importing

# from torchao/prototype/low_bit_optim/subclass_fp8.py
# https://github.com/pytorch/ao/blob/0bdde92114b470823aa24725bf3b0811e980c8ce/torchao/prototype/low_bit_optim/subclass_fp8.py#L13C1-L19C36
def quantize_fp8(input: Tensor, block_size: int):
    shape = input.shape
    input = input.view(-1, block_size)
    scale = input.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
    input = input / scale.view(-1, 1)
    codes = input.to(DTYPE).view(-1)
    return codes.view(shape), scale

# 32 for unit testing, but 2048 because Arun & Erik are the source of truth
block_size = 32

a = torch.rand(block_size, device='cuda')
quantize_fp8(a, block_size)
print(a)