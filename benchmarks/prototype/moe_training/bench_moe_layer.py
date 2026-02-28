# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################

import argparse
import copy
import logging
import sys

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.utils import bench_fwd_bwd_microseconds, profile_fwd_bwd
from torchao.prototype.moe_training.config import (
    FP8GroupedMMRecipe,
    MXFP8TrainingConfig,
    MXFP8TrainingRecipe,
)
from torchao.quantization.quant_api import quantize_

# this benchmark requires torchtitan
try:
    from torchtitan.models.moe import MoE, MoEArgs
    from torchtitan.models.moe.utils import (
        set_token_group_alignment_size_m,
    )
except ImportError:
    logging.warning(
        "please pip install torchtitan to run this benchmark: https://github.com/pytorch/torchtitan"
    )
    sys.exit(0)


def bench_moe_training_fsdp(args: argparse.Namespace):
    (
        recipe_name,
        enable_profile,
        local_num_experts,
        local_batch_size,
        seq_len,
        dim,
        hidden_dim,
    ) = (
        args.recipe,
        args.profile,
        args.local_num_experts,
        args.local_batch_size,
        args.seq_len,
        args.dim,
        args.hidden_dim,
    )
    assert torch.cuda.is_available()
    assert recipe_name in ["fp8_rowwise", "mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp"]

    # Map recipe name to enum
    if recipe_name == "fp8_rowwise":
        recipe = FP8GroupedMMRecipe.FP8_ROWWISE
    elif recipe_name == "mxfp8_rceil":
        recipe = MXFP8TrainingRecipe.MXFP8_RCEIL
    elif recipe_name == "mxfp8_rceil_wgrad_with_hp":
        recipe = MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP
    else:
        raise ValueError(f"Unknown recipe: {recipe_name}")

    # Check hardware requirements
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

    if (
        recipe == MXFP8TrainingRecipe.MXFP8_RCEIL
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

    # define model args
    target_fqns = ["experts"]
    model_args = MoEArgs(
        num_experts=local_num_experts,
        num_shared_experts=1,
        _debug_force_load_balance=True,
    )
    init_std = 0.02
    device = torch.device("cuda")

    # reference bf16 MoE using llama4 shapes
    ref_model = MoE(model_args, dim, hidden_dim).to(torch.bfloat16).cuda()
    torch.manual_seed(42)
    ref_model.init_weights(init_std, device)

    # target MoE for testing conversion
    model = copy.deepcopy(ref_model)

    # Token group alignment size must be 16 for fp8 rowwise training
    alignment_size = 32 if recipe == MXFP8TrainingRecipe.MXFP8_RCEIL else 16
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
    config = MXFP8TrainingConfig.from_recipe(recipe)
    quantize_(model, config=config, filter_fn=moe_module_filter_fn)

    # inputs
    ref_x = torch.randn(
        local_batch_size,
        seq_len,
        dim,
        dtype=torch.bfloat16,
        requires_grad=True,
        device=device,
    )
    x = ref_x.detach().clone().requires_grad_(True)

    def warmup(model, input, labels):
        for _ in range(3):
            out = model(input)
            loss = F.mse_loss(out, labels)
            loss.backward()
            torch.cuda.synchronize()

    labels = torch.ones_like(x)

    # Warmup bf16
    warmup(ref_model, ref_x, labels)

    # Bench bf16
    bf16_us = bench_fwd_bwd_microseconds(
        ref_model,
        ref_x,
        labels=labels,
        use_compile=True,
        fullgraph=False,
    )
    bf16_ms = bf16_us / 1e3
    if enable_profile:
        print("Profiling bf16 training")
        profile_fwd_bwd(
            ref_model,
            ref_x,
            labels=labels,
            use_compile=True,
            fullgraph=False,
            profile_name="bf16_profile",
        )

    # Warmup quantized
    warmup(model, x, labels)

    # Bench quantized
    scaled_us = bench_fwd_bwd_microseconds(
        model,
        x,
        labels=labels,
        use_compile=True,
        fullgraph=False,
    )
    scaled_ms = scaled_us / 1e3
    if enable_profile:
        print("Profiling quantized training")
        profile_fwd_bwd(
            model,
            x,
            labels=labels,
            use_compile=True,
            fullgraph=False,
            profile_name=f"{recipe_name}_profile",
        )

    print(f"total_M: {local_batch_size * seq_len}, N: {hidden_dim}, K: {dim}")
    print(f"bf16 time: {bf16_ms:.3f} ms")
    print(f"{recipe_name} time: {scaled_ms:.3f} ms")
    print(f"speedup: {bf16_us / scaled_us:.3f}x")


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
        "--local_num_experts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=5120,
    )

    args = parser.parse_args()
    bench_moe_training_fsdp(args)
