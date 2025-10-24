# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

# this feature requires CUDA and SM89+
assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

from torchao.prototype.moe_training.conversion_utils import (
    MoEScalingType,
    MoETrainingConfig,
)
from torchao.quantization.quant_api import quantize_

# this example uses torchtitan llama4 MoE, see
try:
    from torchtitan.models.moe import MoE, MoEArgs
    from torchtitan.models.moe.utils import set_token_group_alignment_size_m
except ImportError as e:
    raise ImportError(
        "torchtitan not installed, see installation instructions at https://github.com/pytorch/torchtitan"
    ) from e


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--scaling_type",
    type=str,
    default="fp8_rowwise",
    choices=["fp8_rowwise", "mxfp8"],
)
args = parser.parse_args()


# initialize model
device = torch.device("cuda")
torch.manual_seed(42)
model_args = MoEArgs(num_experts=8, top_k=2, use_grouped_mm=True)
dim = 1024
hidden_dim = dim * 4
model = MoE(model_args, dim, hidden_dim).to(torch.bfloat16).to(device)
init_std = 0.02
model.init_weights(init_std, device)

# module filter function to define which modules to quantize
target_fqns = ["experts"]


def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
    for target_fqn in target_fqns:
        if target_fqn in cur_fqn:
            return True
    return False


if args.scaling_type == "fp8_rowwise":
    config = MoETrainingConfig()
    alignment_size = 16

elif args.scaling_type == "mxfp8":
    config = MoETrainingConfig(scaling_type=MoEScalingType.MXFP8)
    alignment_size = 32

quantize_(model, config=config, filter_fn=moe_module_filter_fn)
set_token_group_alignment_size_m(alignment_size)

# training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(10):
    batch, seq = 8, 2048
    x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )

    # forward pass
    out = model(x)

    # compute loss
    labels = torch.ones_like(out)
    out_loss = F.mse_loss(out, labels)
    print(f"step {step} loss: {out_loss.item()}")

    # backward pass
    out_loss.backward()
    optimizer.step()
