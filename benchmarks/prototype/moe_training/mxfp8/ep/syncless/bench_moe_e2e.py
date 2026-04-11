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
import torch.distributed as dist
from tabulate import tabulate
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from tqdm import tqdm

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "../../../../../../test/prototype/moe_training"
    ),
)
from reference_moe import (
    MoE,
    MoEArgs,
)
from reference_parallel_styles import ExpertParallel

from benchmarks.utils import profile_fn
from torchao.float8.float8_utils import compute_error

from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
    SymmetricMemoryBufferManager,
)
from torchao.prototype.moe_training.ep.syncless.expert_parallel import (
    SynclessExpertParallel,
)
from torchao.prototype.moe_training.ep.syncless.moe import (
    MoEArgs as SynclessMoEArgs,
)
from torchao.prototype.moe_training.ep.syncless.moe import (
    SynclessMXFP8MoE,
)
from torchao.prototype.moe_training.ep.syncless.saved_activations_buffer import (
    SavedActivationsBuffer,
)

device = torch.device("cuda")


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
    ref_fwd_bwd_ms: float
    syncless_fwd_bwd_ms: float


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


def _build_ref_model(config: ExperimentConfig) -> nn.Module:
    """Build standard MoE (bf16), unparallelized."""
    moe_args = MoEArgs(
        num_experts=config.num_experts,
        num_shared_experts=0,
        use_grouped_mm=True,
        _debug_force_load_balance=True,
    )
    model = MoE(moe_args, config.dim, config.hidden_dim).to(torch.bfloat16).cuda()
    model.init_weights(0.02, device)
    return model


def _build_syncless_model(
    config: ExperimentConfig,
    ref_model: nn.Module,
    saved_activations_buffer=None,
) -> nn.Module:
    """Build SynclessMXFP8MoE and copy weights from the ref model."""
    moe_args = SynclessMoEArgs(
        num_experts=config.num_experts,
        num_shared_experts=0,
        use_grouped_mm=True,
        _debug_force_load_balance=True,
    )
    model = (
        SynclessMXFP8MoE(
            moe_args,
            config.dim,
            config.hidden_dim,
            saved_activations_buffer=saved_activations_buffer,
        )
        .to(torch.bfloat16)
        .cuda()
    )

    # Copy weights from ref model so both models start with identical weights.
    # Router gate: identical shape, direct copy.
    model.router.gate.weight.data.copy_(ref_model.router.gate.weight.data)
    # Expert w13: fuse ref's separate w1 and w3 by concatenating along dim=1.
    model.experts.w13.data.copy_(
        torch.cat([ref_model.experts.w1.data, ref_model.experts.w3.data], dim=1)
    )
    # Expert w2: identical shape, direct copy.
    model.experts.w2.data.copy_(ref_model.experts.w2.data)

    return model


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    world_size = dist.get_world_size()
    ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))

    def warmup(fn, n=3):
        for _ in range(n):
            fn()

    NUM_ITERS = 10

    # Shared input for both models (same routing decisions via _debug_force_load_balance)
    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.dim,
        dtype=torch.bfloat16,
        device=device,
    )

    torch.manual_seed(42)
    ref_model = _build_ref_model(config)

    group = ep_mesh.get_group()

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
        group=group,
    )

    # Create saved activations buffer for backward pass
    saved_act_buffer = SavedActivationsBuffer(
        gpu_tokens=max_output_rows,
        cpu_tokens=0,
        dim=config.dim,
        hidden_dim=config.hidden_dim,
        device=device,
    )

    syncless_model = _build_syncless_model(
        config, ref_model, saved_activations_buffer=saved_act_buffer
    )

    parallelize_module(
        module=ref_model.experts,
        device_mesh=ep_mesh,
        parallelize_plan=ExpertParallel(),
    )
    parallelize_module(
        module=syncless_model.experts,
        device_mesh=ep_mesh,
        parallelize_plan=SynclessExpertParallel(buffer_manager=buffer_manager),
    )

    if args.compile:
        ref_model = torch.compile(ref_model)
    warmup(lambda: ref_model(x))

    # Save one output for correctness check
    with torch.no_grad():
        ref_out = ref_model(x)

    # ref model timing (forward)
    torch.cuda.synchronize()
    ref_start_time = time.perf_counter()
    for _ in range(NUM_ITERS):
        ref_model(x)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - ref_start_time) * 1e3 / NUM_ITERS

    def ref_fwd_bwd():
        ref_model.zero_grad()
        y = ref_model(x)
        y.backward(torch.ones_like(y))

    warmup(ref_fwd_bwd)

    torch.cuda.synchronize()
    ref_fb_start = time.perf_counter()
    for _ in range(NUM_ITERS):
        ref_fwd_bwd()
    torch.cuda.synchronize()
    ref_fwd_bwd_ms = (time.perf_counter() - ref_fb_start) * 1e3 / NUM_ITERS

    # Run once more to capture gradients for correctness check
    ref_model.zero_grad()
    ref_out_for_grad = ref_model(x)
    ref_out_for_grad.backward(torch.ones_like(ref_out_for_grad))
    ref_w2_grad = ref_model.experts.w2.grad.to_local().clone()
    ref_w13_grad = torch.cat(
        [ref_model.experts.w1.grad.to_local(), ref_model.experts.w3.grad.to_local()],
        dim=1,
    ).clone()

    if args.profile:

        def ref_batch():
            for _ in range(10):
                ref_model(x)

        profile_fn(
            ref_batch,
            distributed=True,
            profile_name="ref_ep_moe_fwd",
            active_steps=1,
        )

    del ref_model

    # syncless model timing (forward)
    if args.compile:
        syncless_model = torch.compile(syncless_model)
    warmup(lambda: syncless_model(x))

    # Save one output for correctness check
    with torch.no_grad():
        syncless_out = syncless_model(x)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(NUM_ITERS):
        syncless_model(x)
    torch.cuda.synchronize()
    syncless_ms = (time.perf_counter() - start_time) * 1e3 / NUM_ITERS

    # ---- syncless model timing (forward + backward) -----------------------
    def syncless_fwd_bwd():
        syncless_model.zero_grad()
        saved_act_buffer.free_all_py()
        y = syncless_model(x)
        y.backward(torch.ones_like(y))

    warmup(syncless_fwd_bwd)

    torch.cuda.synchronize()
    syncless_fb_start = time.perf_counter()
    for _ in range(NUM_ITERS):
        syncless_fwd_bwd()
    torch.cuda.synchronize()
    syncless_fwd_bwd_ms = (time.perf_counter() - syncless_fb_start) * 1e3 / NUM_ITERS

    if args.profile:

        def syncless_batch():
            for _ in range(10):
                syncless_model(x)

        profile_fn(
            syncless_batch,
            distributed=True,
            profile_name="syncless_ep_moe_fwd",
            active_steps=1,
        )

    # ---- correctness check -----------------------------------------------
    out_sqnr = compute_error(syncless_out, ref_out)

    # Gradient correctness check
    syncless_model.zero_grad()
    saved_act_buffer.free_all_py()
    syncless_out_for_grad = syncless_model(x)
    syncless_out_for_grad.backward(torch.ones_like(syncless_out_for_grad))
    syncless_w2_grad = syncless_model.experts.w2.grad.to_local().clone()
    syncless_w13_grad = syncless_model.experts.w13.grad.to_local().clone()

    w2_grad_sqnr = compute_error(syncless_w2_grad, ref_w2_grad)
    w13_grad_sqnr = compute_error(syncless_w13_grad, ref_w13_grad)

    if dist.get_rank() == 0:
        print(
            f"  Output SQNR for shape "
            f"({config.batch_size}, {config.seq_len}, {config.dim}): "
            f"{out_sqnr.item():.1f} dB"
        )
        print(
            f"  w2 grad SQNR: {w2_grad_sqnr.item():.1f} dB  |  "
            f"w13 grad SQNR: {w13_grad_sqnr.item():.1f} dB"
        )

    del syncless_model, ref_out, syncless_out

    return ExperimentResult(
        ref_ms=ref_ms,
        syncless_ms=syncless_ms,
        ref_fwd_bwd_ms=ref_fwd_bwd_ms,
        syncless_fwd_bwd_ms=syncless_fwd_bwd_ms,
    )


# =========================================================================
# Print results
# =========================================================================


def print_results(experiments: List[Experiment]):
    headers = [
        "shape (B,S,D)",
        "num_experts",
        "num_ranks",
        "ref_fwd_ms",
        "syncless_fwd_ms",
        "fwd_speedup",
        "ref_fwd_bwd_ms",
        "syncless_fwd_bwd_ms",
        "fwd_bwd_speedup",
    ]
    rows = []
    for exp in experiments:
        c, r = exp.config, exp.result
        fwd_speedup = r.ref_ms / r.syncless_ms if r.syncless_ms > 0 else float("inf")
        fb_speedup = (
            r.ref_fwd_bwd_ms / r.syncless_fwd_bwd_ms
            if r.syncless_fwd_bwd_ms > 0
            else float("inf")
        )
        rows.append(
            [
                f"({c.batch_size}, {c.seq_len}, {c.dim})",
                c.num_experts,
                dist.get_world_size(),
                f"{r.ref_ms:.2f}",
                f"{r.syncless_ms:.2f}",
                f"{fwd_speedup:.2f}x",
                f"{r.ref_fwd_bwd_ms:.2f}",
                f"{r.syncless_fwd_bwd_ms:.2f}",
                f"{fb_speedup:.2f}x",
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
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    main(args)
