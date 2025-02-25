# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of bf16 versus float8 with FSDP on. At a high level:
1. start with a reference model, with FSDP on
2. run forward + backward + optim for 2 iterations
3. repeat 2 with float8 enabled (2 iterations needed for delayed scaling)
4. compare outputs and state dict between (2) and (3), should be close
"""

import copy
import os
import warnings

import fire
import pytest

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
)
from torchao.float8.float8_utils import compute_error

torch.manual_seed(0)

B, M, K, N = 8, 8, 32, 32
lr = 0.01
N_ITER = 2


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model(K, N, base_dtype=torch.float32):
    m = nn.Sequential(
        nn.Linear(K, N, dtype=base_dtype),
        nn.ReLU(),
        nn.Linear(N, N, dtype=base_dtype),
        nn.ReLU(),
    )
    return m


# taken from https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# and modified
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    print("args", args)

    emulate, base_dtype, compile = args
    model = get_model(K, N, base_dtype=base_dtype).to(rank)
    model_fp8 = copy.deepcopy(model)

    config = Float8LinearConfig()

    # Note: we only iterate over `scaling_type_weight` because FSDP only interacts
    # with weights.
    convert_to_float8_training(
        model_fp8,
        config=config,
    )

    # To compile FSDP, we need use_orig_params to True
    model = FSDP(model, use_orig_params=True)
    model_fp8 = FSDP(model_fp8, use_orig_params=True)
    # TODO: The following line doesn't work. We should fix it.
    # model = FSDP(torch.compile(model), use_orig_params=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_fp8 = torch.optim.SGD(model_fp8.parameters(), lr=lr)

    # Note: we need two different inputs to properly measure the impact of
    # delayed scaling, before the first input uses dynamic scaling to
    # populate the buffers
    # TODO(future PR): delete ^, since we deleted delayed scaling
    ref_input_global = [
        torch.randn(B, M, K).cuda().to(base_dtype),
        torch.randn(B, M, K).cuda().to(base_dtype),
    ]
    ref_grad_global = [
        torch.randn(B, M, N).cuda().to(base_dtype),
        torch.randn(B, M, N).cuda().to(base_dtype),
    ]
    ref_input_local = []
    ref_grad_local = []

    # basic distributed data sampling
    assert B % world_size == 0
    bsz_local_start = int(rank / world_size * B)
    bsz_local_end = int((rank + 1) / world_size * B)
    for idx in range(N_ITER):
        ref_input_local.append(
            ref_input_global[idx][bsz_local_start:bsz_local_end].to(rank)
        )
        ref_grad_local.append(
            ref_grad_global[idx][bsz_local_start:bsz_local_end].to(rank)
        )

    def forward_backward(model, optim, is_fp8, i):
        optim.zero_grad()
        y_local = model(ref_input_local[i])
        y_local.backward(ref_grad_local[i])
        optim.step()
        return y_local

    for i in range(N_ITER):
        # We first run one iteration without compile, as a workaround to compile float8 layer.
        # In the first iter, float8 layers go to the branches of "self.is_amax_initialized == False"
        # After that, float8 layers go the the branches of "self.is_amax_initialized == True"
        # TODO: Need to fix compile to run wihtout this workaround.
        if i == 1 and compile:
            model = torch.compile(model)
            model_fp8 = torch.compile(model_fp8)
        y_local = forward_backward(model, optimizer, is_fp8=False, i=i)
        y_local_fp8 = forward_backward(model_fp8, optimizer_fp8, is_fp8=True, i=i)
        local_sqnr = compute_error(y_local, y_local_fp8)  # noqa: F841

    # get global y
    y_global = [
        torch.zeros(*y_local.shape, dtype=base_dtype).to(rank)
        for r in range(world_size)
    ]
    dist.all_gather(y_global, y_local)
    y_global = torch.cat(y_global, dim=0)
    y_global_fp8 = [
        torch.zeros(*y_local_fp8.shape, dtype=base_dtype).to(rank)
        for r in range(world_size)
    ]
    dist.all_gather(y_global_fp8, y_local_fp8)
    y_global_fp8 = torch.cat(y_global_fp8, dim=0)
    if rank == 0:
        sqnr = compute_error(y_global, y_global_fp8)
        assert sqnr > 15.0, f"SQNR of {sqnr} is too low"

    # get global state dict
    # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
    dist.barrier()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    with FSDP.state_dict_type(model_fp8, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_fp8 = model_fp8.state_dict()
    if rank == 0:
        for k, v1 in cpu_state.items():
            v2 = cpu_state_fp8[k]
            v1, v2 = v1.cpu(), v2.cpu()
            sqnr = compute_error(v1, v2)
            assert sqnr > 15.0, f"SQNR of {sqnr} is too low, k: {k}, v1: {v1}, v2: {v2}"

    cleanup()


def run(compile_fsdp: bool = False):
    base_dtype = torch.bfloat16

    emulate = False
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, running in emulation_mode")
        emulate = True
    elif torch.cuda.get_device_capability() < (8, 9):
        warnings.warn(
            f"CUDA capability {torch.cuda.get_device_capability()} < (8.9), running in emulation mode"
        )
        emulate = True

    WORLD_SIZE = torch.cuda.device_count()
    args = (emulate, base_dtype, compile_fsdp)
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    fire.Fire(run)
