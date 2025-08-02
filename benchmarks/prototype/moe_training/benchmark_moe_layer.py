# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these benchmarks, use the following command:
#
# torchrun --nproc-per-node=8 --local-ranks-filter=0 torchao/prototype/moe_training/benchmarks/benchmark_moe_layer.py
#
#######################################################################

import argparse
import copy
import os
import statistics
from time import perf_counter_ns

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.nn import functional as F

# this feature requires CUDA and SM89+
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip(
        "CUDA not available or compute capability < 8.9", allow_module_level=True
    )

from torchao.prototype.moe_training.conversion_utils import MoETrainingConfig
from torchao.quantization.quant_api import quantize_

# this test requires torchtitan
try:
    from torchtitan.experiments.llama4.infra.expert_parallel import (
        set_token_group_alignment_size_m,
    )
    from torchtitan.experiments.llama4.model.args import TransformerModelArgs
    from torchtitan.experiments.llama4.model.moe import MoE
except ImportError:
    pytest.skip(
        "torchtitan not installed, skipping MoE tests.", allow_module_level=True
    )


def bench_moe_float8_training_fsdp(enable_profile=False):
    assert torch.cuda.is_available()

    # setup distributed for fsdp
    setup_distributed()

    # define model args
    target_fqns = ["experts"]
    model_args = TransformerModelArgs(
        moe_enabled=True,
        num_experts=16,
        dim=5120,
    )
    init_std = 0.02
    device = torch.device("cuda")

    # reference bf16 MoE
    ref_model = MoE(model_args).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
    ref_model.init_weights(init_std, device)

    # target MoE for testing conversion
    model = copy.deepcopy(ref_model)

    # assert starting params are identical for both models
    for param1, param2 in zip(model.parameters(), ref_model.parameters()):
        assert torch.equal(param1, param2)

    # convert MoE to float8 training
    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in target_fqns:
            if target_fqn in cur_fqn:
                return True
        return False

    # quantize test model
    config = MoETrainingConfig()
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # FSDP2
    fully_shard(model)
    fully_shard(ref_model)

    # inputs (llama4 shapes)
    batch, seq, dim = 1, 8192, 5120
    ref_x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )
    x = ref_x.detach().clone().requires_grad_(True)

    def bench_fn_microseconds(model, input):
        labels = torch.ones_like(input)
        times = []
        for _ in range(10):
            start_ns = perf_counter_ns()
            out = model(input)
            loss = F.mse_loss(out, labels)
            loss.backward()
            torch.cuda.synchronize()
            end_ns = perf_counter_ns()
            duration_us = (end_ns - start_ns) / 1000
            times.append(duration_us)
        return statistics.median(times)

    def profile_fn(model, input, profile_name="profile"):
        # Only profile on rank 0
        if torch.distributed.get_rank() == 0:
            labels = torch.ones_like(input)
            wait, warmup, active = 1, 3, 1
            total_steps = wait + warmup + active
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=0
                ),
                record_shapes=True,
                with_stack=True,
            ) as prof:
                for _ in range(total_steps):
                    out = model(input)
                    loss = F.mse_loss(out, labels)
                    loss.backward()
                    prof.step()

            # Save profiler results
            prof.export_chrome_trace(f"{profile_name}.json")
            print(f"Saved: {profile_name}.json")

    # Compile models
    ref_model = torch.compile(ref_model, fullgraph=False)
    model = torch.compile(model, fullgraph=False)

    print("Benchmarking MoE with FSDP2 using bf16 training")
    bf16_us = bench_fn_microseconds(ref_model, ref_x)
    print(f"bf16 time: {bf16_us} us")
    if enable_profile:
        print("Profiling bf16 model")
        profile_fn(ref_model, ref_x, profile_name="bf16_profile")

    # Token group alignment size must be 16 for fp8 rowwise training
    set_token_group_alignment_size_m(16)

    print("Benchmarking MoE with FSDP2 using fp8 rowwise training")
    fp8_us = bench_fn_microseconds(model, x)
    print(f"fp8 time: {fp8_us} us")
    if enable_profile:
        print("Profiling fp8 model")
        profile_fn(model, x, profile_name="fp8_profile")

    dist.destroy_process_group()


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MoE layer with FSDP2")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiling and save results to file",
    )
    args = parser.parse_args()
    bench_moe_float8_training_fsdp(enable_profile=args.profile)
