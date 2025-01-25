# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable

import fire
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchao.float8.config import CastConfig, Float8LinearConfig, ScalingType
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
    sync_float8_amax_and_scale_history,
)

torch.manual_seed(0)

# TODO: Add more shapes for the benchmark
B, M, K, N = 32, 1024, 1024, 1024
lr = 0.01

config = Float8LinearConfig(
    cast_config_input=CastConfig(scaling_type=ScalingType.DELAYED),
    cast_config_weight=CastConfig(scaling_type=ScalingType.DELAYED),
    cast_config_grad_output=CastConfig(scaling_type=ScalingType.DELAYED),
)


def benchmark_torch_function_in_microseconds(
    func: Callable,
    *args,
    **kwargs,
) -> float:
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.blocked_autorange().median * 1e6


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_model(K, N, is_fp8, base_dtype=torch.float32):
    modules = [
        nn.Linear(K, N, dtype=base_dtype),
        nn.ReLU(),
    ]
    N_LAYERS = 20
    # N linear layers
    for _ in range(N_LAYERS - 1):
        modules.append(nn.Linear(N, N, dtype=base_dtype))
        modules.append(nn.ReLU())
    m = nn.Sequential(*modules)
    if is_fp8:
        convert_to_float8_training(
            m,
            config=config,
        )
    return m


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    base_dtype, input_global, compile = args

    # basic distributed data sampling
    assert B % world_size == 0
    bsz_local_start = int(rank / world_size * B)
    bsz_local_end = int((rank + 1) / world_size * B)
    input_tensor = input_global[bsz_local_start:bsz_local_end].to(rank)

    fp8_model = get_model(K, N, is_fp8=True, base_dtype=base_dtype).to(rank)
    # Need use_orig_params=True to compile FSDP
    fp8_model = FSDP(fp8_model, use_orig_params=True)
    fp8_optimizer = torch.optim.SGD(fp8_model.parameters(), lr=lr * world_size)

    # Run one iteration to make compile work, see experiments doc for more context of this issue.
    fp8_optimizer.zero_grad()
    y_local = fp8_model(input_tensor)
    y_local.sum().backward()
    fp8_optimizer.step()
    sync_float8_amax_and_scale_history(fp8_model)

    sync_float8_func = sync_float8_amax_and_scale_history
    if compile:
        # TODO: Need to fix issues with compile
        fp8_model = torch.compile(fp8_model)
        sync_float8_func = torch.compile(sync_float8_amax_and_scale_history)

    def float8_forw_backward():
        fp8_optimizer.zero_grad()
        y_local = fp8_model(input_tensor)
        y_local.sum().backward()
        fp8_optimizer.step()
        sync_float8_func(fp8_model)

    ref_model = get_model(K, N, is_fp8=False, base_dtype=base_dtype).to(rank)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=lr * world_size)
    if compile:
        ref_model = torch.compile(ref_model)

    ref_model = FSDP(ref_model, use_orig_params=True)

    def ref_forw_backward():
        ref_optimizer.zero_grad()
        ref_model(input_tensor).sum().backward()
        ref_optimizer.step()

    def run_n_iterations(n, fn):
        for _ in range(n):
            fn()
        # make sure training is done on all ranks
        dist.barrier()

    # warmup
    run_n_iterations(50, ref_forw_backward)
    run_n_iterations(50, float8_forw_backward)

    N_ITER = 50
    ref_time = (
        benchmark_torch_function_in_microseconds(
            run_n_iterations, N_ITER, ref_forw_backward
        )
        * 1e-6
        / N_ITER
    )
    float8_time = (
        benchmark_torch_function_in_microseconds(
            run_n_iterations, N_ITER, float8_forw_backward
        )
        * 1e-6
        / N_ITER
    )

    if rank == 0:
        print("ref_time", ref_time)
        print("float8_time", float8_time)
        print("float8 speedup", ref_time / float8_time)

    cleanup()


def run(compile: bool):
    base_dtype = torch.bfloat16
    WORLD_SIZE = torch.cuda.device_count()
    print(f"{base_dtype = }")
    print(f"{compile = }")
    print(f"{WORLD_SIZE = }")

    # generate input data
    ref_input = torch.randn(B, M, K).cuda().to(base_dtype)
    # run fsdp model
    args = (base_dtype, ref_input, compile)
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


# Usgae:
# CUDA_VISIBLE_DEVICES=0,1 python benchmarks/bench_multi_gpu.py
if __name__ == "__main__":
    fire.Fire(run)
