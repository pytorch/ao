# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test autocast + torch.compile + FSDP + Float8Linear
"""

import os
import unittest
import warnings

import fire

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise unittest.SkipTest("Unsupported PyTorch version")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
)

torch.manual_seed(0)

B, M, K, N = 8, 8, 32, 32
lr = 0.01
N_ITER = 1


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model(K, N, is_fp8, emulate, base_dtype=torch.float32):
    # composability of torch.compile + FSDP + autocast + Float8Linear
    # as fo 2023-12-30

    # without any changes to the Float8Linear, we get this error:
    # https://gist.github.com/vkuzo/3bcb81806cc92f99ac0b9c5fdf287730

    # if we initialize Float8Linear with is_amax_initialized=True and
    # amax_and_scale_synced=True, we get
    # https://gist.github.com/vkuzo/ed8e168fd9f7463f1fce34301334ab55
    # to get around this, we can disable amax init
    config = Float8LinearConfig(
        emulate=emulate,
    )

    m = nn.Sequential(
        nn.Linear(K, N, dtype=base_dtype),
        nn.ReLU(),
    )
    convert_to_float8_training(
        m,
        config=config,
    )
    return m


# taken from https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# and modified
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    (emulate,) = args

    # finally, if we remove the usage of self.bias_dtype, then
    # things work e2e. Note that FSDP does not support full-graph compile
    # regardless of float8.

    model = get_model(K, N, is_fp8=True, emulate=emulate, base_dtype=torch.bfloat16).to(
        rank
    )

    # To compile FSDP, we need use_orig_params to True
    model = FSDP(model, use_orig_params=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr * world_size)
    input_local = torch.randn(B, M, K, N, device="cuda")

    model = torch.compile(model)

    for _iter in range(N_ITER):
        optimizer.zero_grad()
        with torch.autocast("cuda"):
            y_local = model(input_local)
        y_local.sum().backward()
        optimizer.step()

    print("done!")
    cleanup()


def run():
    emulate = False
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, running in emulation_mode", stacklevel=2)
        emulate = True
    elif torch.cuda.get_device_capability() < (9, 0):
        warnings.warn(
            f"CUDA capability {torch.cuda.get_device_capability()} < (9.0), running in emulation mode",
            stacklevel=2,
        )
        emulate = True

    WORLD_SIZE = torch.cuda.device_count()
    args = (emulate,)
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    fire.Fire(run)
