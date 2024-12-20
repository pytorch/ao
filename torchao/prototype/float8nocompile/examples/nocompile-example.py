import torch
import torch.nn as nn

from torchao.prototype.float8nocompile.float8nocompile_linear_utils import (
    convert_to_float8_nocompile_training,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")

# create model and sample input
m = (
    nn.Sequential(
        nn.Linear(32, 32, bias=False),
    )
    .bfloat16()
    .cuda()
)
x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

# convert specified `torch.nn.Linear` modules to `Float8Linear`
print("calling convert_to_float8_nocompile_training")
convert_to_float8_nocompile_training(m)
print("finished convert_to_float8_nocompile_training")

for i in range(10):
    print(f"step {i}")
    optimizer.zero_grad()
    y = m(x)
    y.sum().backward()
    optimizer.step()
