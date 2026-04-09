# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# End-to-end MoE benchmark comparing:
#   1. Standard ExpertParallel (bf16 all-to-all + permute/unpermute + bf16 grouped MM, requires d2h syncs)
#   2. Syncless ExpertParallel (MXFP8 symm-mem dispatch/combine + MXFP8 grouped MM, no d2h syncs)
#
# To run:
#   torchrun --nproc-per-node=2 --local-ranks-filter=0 <path to file>
#
#######################################################################
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.distributed as dist
from tabulate import tabulate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from tqdm import tqdm

# -- reference MoE and parallel styles (from test/) -----------------------
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "../../../../../../test/prototype/moe_training"
    ),
)
from benchmarks.utils import profile_fn
from reference_moe import (
    MoE,
    MoEArgs,
    set_token_group_alignment_size_m,
)
from reference_parallel_styles import ExpertParallel

# -- syncless EP -----------------------------------------------------------
from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
    SymmetricMemoryBufferManager,
)
from torchao.prototype.moe_training.ep.syncless.expert_parallel import (
    SynclessExpertParallel,
)
from torchao.prototype.moe_training.ep.syncless.moe import (
    MoEArgs as SynclessMoEArgs,
    SynclessMXFP8MoE,
)

device = torch.device("cuda")


# =========================================================================
# Experiment config / result
# =========================================================================
@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    seq_len: int
    dim: int
    hidden_dim: int
    num_experts: int


@dataclass(frozen=True)
class ExperimentResult:
    ref_ms: float
    syncless_ms: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            batch_size=1, seq_len=8192, dim=7168, hidden_dim=2048, num_experts=8
        ),
    ]


# =========================================================================
# Build models
# =========================================================================


def _build_ref_model(config: ExperimentConfig, ep_mesh) -> nn.Module:
    """Standard MoE + ExpertParallel (bf16 all-to-all + bf16 grouped MM)."""
    moe_args = MoEArgs(
        num_experts=config.num_experts,
        num_shared_experts=0,
        use_grouped_mm=True,
        _debug_force_load_balance=True,
    )
    model = MoE(moe_args, config.dim, config.hidden_dim).to(torch.bfloat16).cuda()
    model.init_weights(0.02, device)

    set_token_group_alignment_size_m(32)
    parallelize_module(
        module=model.experts,
        device_mesh=ep_mesh,
        parallelize_plan=ExpertParallel(),
    )
    return model


def _build_syncless_model(
    config: ExperimentConfig,
    ep_mesh,
    buffer_manager: SymmetricMemoryBufferManager,
) -> nn.Module:
    """SynclessMXFP8MoE + SynclessExpertParallel."""
    moe_args = SynclessMoEArgs(
        num_experts=config.num_experts,
        num_shared_experts=0,
        use_grouped_mm=True,
        _debug_force_load_balance=True,
    )
    model = (
        SynclessMXFP8MoE(moe_args, config.dim, config.hidden_dim)
        .to(torch.bfloat16)
        .cuda()
    )
    model.init_weights(0.02, device)

    parallelize_module(
        module=model.experts,
        device_mesh=ep_mesh,
        parallelize_plan=SynclessExpertParallel(buffer_manager=buffer_manager),
    )
    return model


# =========================================================================
# Run experiment
# =========================================================================


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    world_size = dist.get_world_size()
    ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))

    def warmup(fn, n=3):
        for _ in range(n):
            fn()

    NUM_ITERS = 10

    # ---- reference model -------------------------------------------------
    torch.manual_seed(42)
    ref_model = _build_ref_model(config, ep_mesh)
    x_ref = torch.randn(
        config.batch_size,
        config.seq_len,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
    )
    warmup(lambda: ref_model(x_ref))

    torch.cuda.synchronize()
    ref_start_time = time.perf_counter()
    for _ in range(NUM_ITERS):
        ref_model(x_ref)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - ref_start_time) * 1e3 / NUM_ITERS

    if args.profile:

        def ref_batch():
            for _ in range(10):
                ref_model(x_ref)

        profile_fn(
            ref_batch,
            distributed=True,
            profile_name="ref_ep_moe_fwd",
            active_steps=3,
        )

    del ref_model

    # ---- syncless model ---------------------------------------------------
    torch.manual_seed(42)

    buffer_manager = SymmetricMemoryBufferManager()
    total_tokens = config.batch_size * config.seq_len
    max_output_rows = total_tokens * world_size
    buffer_manager.preallocate_buffers(
        max_output_rows_per_rank=max_output_rows,
        data_shape=(config.dim,),
        scales_shape=(config.dim // 32,),
        data_dtype=torch.float8_e4m3fn,
        scales_dtype=torch.uint8,
        device=device,
    )

    syncless_model = _build_syncless_model(config, ep_mesh, buffer_manager)
    x_sync = torch.randn(
        config.batch_size,
        config.seq_len,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
    )
    warmup(lambda: syncless_model(x_sync))

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(NUM_ITERS):
        syncless_model(x_sync)
    torch.cuda.synchronize()
    syncless_ms = (time.perf_counter() - start_time) * 1e3 / NUM_ITERS

    if args.profile:

        def syncless_batch():
            for _ in range(10):
                syncless_model(x_sync)

        profile_fn(
            syncless_batch,
            distributed=True,
            profile_name="syncless_ep_moe_fwd",
            active_steps=3,
        )

    del syncless_model

    return ExperimentResult(ref_ms=ref_ms, syncless_ms=syncless_ms)


# =========================================================================
# Print results
# =========================================================================


def print_results(experiments: List[Experiment]):
    headers = [
        "shape (B,S,D)",
        "num_experts",
        "num_ranks",
        "ref_EP_ms",
        "syncless_EP_ms",
        "speedup",
    ]
    rows = []
    for exp in experiments:
        c, r = exp.config, exp.result
        speedup = r.ref_ms / r.syncless_ms if r.syncless_ms > 0 else float("inf")
        rows.append(
            [
                f"({c.batch_size}, {c.seq_len}, {c.dim})",
                c.num_experts,
                dist.get_world_size(),
                f"{r.ref_ms:.2f}",
                f"{r.syncless_ms:.2f}",
                f"{speedup:.2f}x",
            ]
        )
    print("\n" + "=" * 100)
    print("MoE END-TO-END BENCHMARK: Standard EP vs Syncless MXFP8 EP")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


# =========================================================================
# Main
# =========================================================================


def main(args: argparse.Namespace):
    torch.manual_seed(123)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    print_results(results)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    main(args)
