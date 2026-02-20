# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these benchmarks, use the following command:
#
# torchrun --nproc-per-node=8 --local-ranks-filter=0 benchmarks/prototype/moe_training/benchmark_moe_layer_fsdp.py
#
#######################################################################

import argparse
import copy
import logging
import os

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.nn import functional as F

from benchmarks.utils import bench_fwd_bwd_microseconds, profile_fwd_bwd
from torchao.prototype.fp8_grouped_mm.config import FP8GroupedMMRecipe
from torchao.prototype.mx_formats.grouped_mm.config import (
    MXFP8GroupedMMConfig,
    MXFP8GroupedMMRecipe,
)
from torchao.quantization.quant_api import quantize_

# this benchmark requires torchtitan
try:
    from torchtitan.models.moe import MoE, MoEArgs
    from torchtitan.models.moe.utils import (
        set_token_group_alignment_size_m,
    )
except ImportError:
    pytest.skip(
        "torchtitan not installed, skipping MoE tests.", allow_module_level=True
    )


def bench_moe_training_fsdp(recipe_name: str, enable_profile: bool, use_compile: bool):
    assert torch.cuda.is_available()
    assert recipe_name in ["fp8_rowwise", "mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp"]
    # Map recipe names to enums
    if recipe_name.upper() == "fp8_rowwise":
        recipe = FP8GroupedMMRecipe.FP8_ROWWISE
    elif recipe_name.upper() == "mxfp8_rceil":
        recipe = MXFP8GroupedMMRecipe.MXFP8_RCEIL
    elif recipe_name.upper() == "mxfp8_rceil_wgrad_with_hp":
        recipe = MXFP8GroupedMMRecipe.MXFP8_RCEIL_WGRAD_WITH_HP
    else:
        raise ValueError(f"Unknown recipe: {recipe_name}")
    if (
        recipe == FP8GroupedMMRecipe.FP8_ROWWISE
        and torch.cuda.get_device_capability()
        != (
            9,
            0,
        )
    ):
        logging.warning(
            f"Skipping FP8 rowwise benchmarks, only supported on compute capability 9.0 and found {torch.cuda.get_device_capability()}"
        )
        return

    elif (
        recipe == MXFP8GroupedMMRecipe.MXFP8_RCEIL
        and torch.cuda.get_device_capability()
        != (
            10,
            0,
        )
    ):
        logging.warning(
            f"Skipping MXFP8 benchmarks, only supported on compute capability 10.0 and found {torch.cuda.get_device_capability()}"
        )
        return

    # setup distributed for fsdp
    setup_distributed()

    # define model args
    target_fqns = ["experts"]
    model_args = MoEArgs(
        num_experts=16,
        num_shared_experts=1,
        _debug_force_load_balance=True,
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
    alignment_size = 32 if recipe == MXFP8GroupedMMRecipe.MXFP8_RCEIL else 16
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
    config = MXFP8GroupedMMConfig.from_recipe(recipe)
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
    parser.add_argument(
        "--recipe",
        type=str,
        help="[fp8_rowwise, rceil, rceil_wgrad_with_hp]",
        required=True,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="use torch.compile",
    )
    args = parser.parse_args()
    bench_moe_training_fsdp(
        recipe_name=args.recipe,
        enable_profile=args.profile,
        use_compile=args.compile,
    )
