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

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.nn import functional as F

from benchmarks.prototype.moe_training.utils import (
    bench_fwd_bwd_microseconds,
    profile_fwd_bwd,
)

# this feature requires CUDA and SM89+
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip(
        "CUDA not available or compute capability < 8.9", allow_module_level=True
    )

from torchao.prototype.moe_training.conversion_utils import (
    MoEScalingType,
    MoETrainingConfig,
)
from torchao.quantization.quant_api import quantize_

# this benchmark requires torchtitan
try:
    from torchtitan.distributed.expert_parallel import (
        set_token_group_alignment_size_m,
    )
    from torchtitan.models.moe import MoE, MoEArgs
except ImportError:
    pytest.skip(
        "torchtitan not installed, skipping MoE tests.", allow_module_level=True
    )


def bench_moe_float8_training_fsdp(
    recipe_name: str, enable_profile: bool, use_compile: bool
):
    assert torch.cuda.is_available()
    assert recipe_name in ["fp8_rowwise", "mxfp8"]
    recipe = MoEScalingType[recipe_name.upper()]

    # setup distributed for fsdp
    setup_distributed()

    # define model args
    target_fqns = ["experts"]
    model_args = MoEArgs(
        num_experts=16,
    )
    init_std = 0.02
    device = torch.device("cuda")

    # reference bf16 MoE using llama4 shapes
    dim, hidden_dim = 5120, 8192
    ref_model = MoE(model_args, dim, hidden_dim).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
    ref_model.init_weights(init_std, device)

    # target MoE for testing conversion
    model = copy.deepcopy(ref_model)

    # Token group alignment size must be 16 for fp8 rowwise training
    alignment_size = 32 if recipe == MoEScalingType.MXFP8 else 16
    set_token_group_alignment_size_m(alignment_size)

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
    config = MoETrainingConfig(scaling_type=recipe)
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # FSDP2
    fully_shard(model)
    fully_shard(ref_model)

    # inputs (llama4 shapes)
    batch, seq = 1, 16640
    ref_x = torch.randn(
        batch, seq, dim, dtype=torch.bfloat16, requires_grad=True, device=device
    )
    x = ref_x.detach().clone().requires_grad_(True)

    def warmup(model, input):
        for _ in range(3):
            out = model(input)
            loss = F.mse_loss(out, torch.ones_like(out))
            loss.backward()
            torch.cuda.synchronize()

    labels = torch.ones_like(x)

    # TODO: bench with fullgraph=True if/when it is supported
    bf16_us = bench_fwd_bwd_microseconds(
        ref_model,
        ref_x,
        labels=labels,
        use_compile=use_compile,
        fullgraph=False,
    )
    print(f"BF16 time: {bf16_us} us")
    if enable_profile:
        print("Profiling bf16 training")
        profile_fwd_bwd(ref_model, ref_x, labels=labels, profile_name="bf16_profile")

    scaled_us = bench_fwd_bwd_microseconds(
        model,
        x,
        labels=labels,
        use_compile=use_compile,
        fullgraph=False,
    )
    print(f"Scaled time: {scaled_us} us")
    if enable_profile:
        print("Profiling quantized training")
        profile_fwd_bwd(model, x, labels=labels, profile_name=f"{recipe_name}_profile")

    print(f"Speedup: {bf16_us / scaled_us:.3f}x")
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
    parser.add_argument("--recipe", type=str, help="[fp8_rowwise, mxfp8]")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="use torch.compile",
    )
    args = parser.parse_args()
    bench_moe_float8_training_fsdp(
        recipe_name=args.recipe,
        enable_profile=args.profile,
        use_compile=args.compile,
    )
