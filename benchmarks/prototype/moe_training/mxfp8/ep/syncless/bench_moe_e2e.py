# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# End-to-end MoE benchmark comparing:
#   1. Standard ExpertParallel (bf16 all-to-all + permute/unpermute + bf16 grouped MM, requires d2h syncs)
#   2. TorchAO MXFP8 ExpertParallel (quantize_ API + TorchAOExpertParallel, MXFP8 grouped MM, requires d2h syncs)
#   3. Syncless ExpertParallel (MXFP8 symm-mem dispatch/combine + MXFP8 grouped MM, no d2h syncs)
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
from reference_parallel_styles import ExpertParallel, TorchAOExpertParallel

from benchmarks.utils import profile_fn
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.config import (
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)
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
from torchao.quantization.quant_api import quantize_
from torchao.testing.training.roofline_utils import get_specs

device = torch.device("cuda")


def _measure_peak_memory_fwd_bwd(fn) -> float:
    """Run fn() once and return peak GPU memory allocated in MiB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    seq_len: int
    dim: int
    hidden_dim: int
    num_experts: int


@dataclass(frozen=True)
class ExperimentResult:
    ref_fwd_bwd_ms: float
    torchao_mxfp8_fwd_bwd_ms: float
    syncless_fwd_bwd_ms: float
    ref_mfu_pct: float
    torchao_mxfp8_mfu_pct: float
    syncless_mfu_pct: float
    ref_peak_mem_mib: float
    torchao_mxfp8_peak_mem_mib: float
    syncless_peak_mem_mib: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def _compute_mfu(
    config: ExperimentConfig,
    top_k: int,
    time_ms: float,
    approach: str,
) -> float:
    """Compute MFU % for a given approach, accounting for mixed bf16/fp8 precision.

    Expert FFN per rank (fwd+bwd) has two linear layers:
      w13 layer (fused w1+w3): 3 GEMMs, each 2*M*dim*2H -> 12*M*dim*H total
      w2 layer:                3 GEMMs, each 2*M*H*dim  ->  6*M*H*dim total

    Precision breakdown by approach:
      ref:      all 6 GEMMs bf16
      torchao:  all 6 GEMMs mxfp8
      syncless: 5 GEMMs mxfp8, 1 GEMM bf16 (w2 wgrad = grad_out.T @ h)
    """
    specs = get_specs()
    bf16_peak = specs["bf16_peak_tops"]
    fp8_peak = specs["fp8_peak_tops"]

    # Tokens processed by this rank's local experts (load-balanced)
    M = config.batch_size * config.seq_len * top_k
    D = config.dim
    H = config.hidden_dim

    w13_flops = 12 * M * D * H  # 3 GEMMs of (M, D) x (D, 2H)
    w2_flops = 6 * M * H * D  # 3 GEMMs of (M, H) x (H, D)
    total_flops = w13_flops + w2_flops  # = 18 * M * H * D

    if approach == "ref":
        # All GEMMs bf16
        theoretical_peak_time_s = total_flops / bf16_peak
    elif approach == "torchao":
        # All expert GEMMs mxfp8
        theoretical_peak_time_s = total_flops / fp8_peak
    elif approach == "syncless":
        # w13: all 3 GEMMs mxfp8 (12*M*D*H)
        # w2: fwd + dgrad mxfp8 (4*M*H*D), wgrad bf16 (2*M*H*D)
        fp8_flops = w13_flops + 4 * M * H * D
        bf16_flops = 2 * M * H * D
        theoretical_peak_time_s = fp8_flops / fp8_peak + bf16_flops / bf16_peak
    else:
        raise ValueError(f"Unknown approach: {approach}")

    actual_time_s = time_ms / 1000.0
    return theoretical_peak_time_s / actual_time_s * 100.0


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            batch_size=4, seq_len=4096, dim=7168, hidden_dim=2048, num_experts=4
        ),
    ]


def _build_ref_model(config: ExperimentConfig) -> nn.Module:
    """Build standard MoE (bf16), unparallelized."""
    moe_args = MoEArgs(
        num_experts=config.num_experts,
        top_k=4,
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
        top_k=4,
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
    top_k = 4  # must match the top_k used in _build_ref_model / _build_syncless_model
    max_output_rows = total_tokens * top_k
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

    # Peak memory measurement
    ref_peak_mem_mib = _measure_peak_memory_fwd_bwd(ref_fwd_bwd)

    if args.profile:

        def ref_batch():
            for _ in range(10):
                ref_fwd_bwd()

        profile_fn(
            ref_batch,
            distributed=True,
            profile_name="ref_ep_moe_fwd",
            active_steps=1,
        )

    del ref_model

    # =========================================================================
    # Approach 2: TorchAO MXFP8 quantize_ API + TorchAOExpertParallel
    # =========================================================================
    torch.manual_seed(42)
    torchao_model = _build_ref_model(config)

    # Apply MXFP8 quantization via quantize_ API
    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        return "experts" in cur_fqn

    mxfp8_config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_RCEIL)
    quantize_(torchao_model, config=mxfp8_config, filter_fn=moe_module_filter_fn)

    # Apply TorchAOExpertParallel (EP with permute_and_pad for MXFP8 alignment)
    parallelize_module(
        module=torchao_model.experts,
        device_mesh=ep_mesh,
        parallelize_plan=TorchAOExpertParallel(pad_multiple=32),
    )

    if args.compile:
        torchao_model = torch.compile(torchao_model)
    warmup(lambda: torchao_model(x))

    def torchao_fwd_bwd():
        torchao_model.zero_grad()
        y = torchao_model(x)
        y.backward(torch.ones_like(y))

    warmup(torchao_fwd_bwd)

    torch.cuda.synchronize()
    torchao_fb_start = time.perf_counter()
    for _ in range(NUM_ITERS):
        torchao_fwd_bwd()
    torch.cuda.synchronize()
    torchao_mxfp8_fwd_bwd_ms = (
        (time.perf_counter() - torchao_fb_start) * 1e3 / NUM_ITERS
    )

    # Peak memory measurement
    torchao_mxfp8_peak_mem_mib = _measure_peak_memory_fwd_bwd(torchao_fwd_bwd)

    if args.profile:

        def torchao_batch():
            for _ in range(10):
                torchao_fwd_bwd()

        profile_fn(
            torchao_batch,
            distributed=True,
            profile_name="torchao_mxfp8_ep_moe",
            active_steps=1,
        )

    del torchao_model

    # =========================================================================
    # Approach 3: Syncless ExpertParallel
    # =========================================================================

    if args.compile:
        syncless_model = torch.compile(syncless_model)
    warmup(lambda: syncless_model(x))

    # Save one output for correctness check
    with torch.no_grad():
        syncless_out = syncless_model(x)

    # ---- syncless model timing (forward + backward) -----------------------
    def syncless_fwd_bwd():
        syncless_model.zero_grad()
        saved_act_buffer.free_all()
        y = syncless_model(x)
        y.backward(torch.ones_like(y))

    warmup(syncless_fwd_bwd)

    torch.cuda.synchronize()
    syncless_fb_start = time.perf_counter()
    for _ in range(NUM_ITERS):
        syncless_fwd_bwd()
    torch.cuda.synchronize()
    syncless_fwd_bwd_ms = (time.perf_counter() - syncless_fb_start) * 1e3 / NUM_ITERS

    # Peak memory measurement (clear buffer first)
    saved_act_buffer.free_all()
    syncless_peak_mem_mib = _measure_peak_memory_fwd_bwd(syncless_fwd_bwd)

    if args.profile:

        def syncless_batch():
            for _ in range(10):
                syncless_fwd_bwd()

        profile_fn(
            syncless_batch,
            distributed=True,
            profile_name="syncless_ep_moe_fwd",
            active_steps=1,
        )

    # ---- correctness check

    # ---- correctness check -----------------------------------------------
    out_sqnr = compute_error(syncless_out, ref_out)

    # Gradient correctness check
    syncless_model.zero_grad()
    saved_act_buffer.free_all()
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

    # Compute MFU %
    ref_mfu = _compute_mfu(config, top_k, ref_fwd_bwd_ms, "ref")
    torchao_mfu = _compute_mfu(config, top_k, torchao_mxfp8_fwd_bwd_ms, "torchao")
    syncless_mfu = _compute_mfu(config, top_k, syncless_fwd_bwd_ms, "syncless")

    return ExperimentResult(
        ref_fwd_bwd_ms=ref_fwd_bwd_ms,
        torchao_mxfp8_fwd_bwd_ms=torchao_mxfp8_fwd_bwd_ms,
        syncless_fwd_bwd_ms=syncless_fwd_bwd_ms,
        ref_mfu_pct=ref_mfu,
        torchao_mxfp8_mfu_pct=torchao_mfu,
        syncless_mfu_pct=syncless_mfu,
        ref_peak_mem_mib=ref_peak_mem_mib,
        torchao_mxfp8_peak_mem_mib=torchao_mxfp8_peak_mem_mib,
        syncless_peak_mem_mib=syncless_peak_mem_mib,
    )


# =========================================================================
# Print results
# =========================================================================


def print_results(experiments: List[Experiment]):
    headers = [
        "shape (B,S,D)",
        "num_experts",
        "num_ranks",
        "ref_fwd_bwd_ms",
        "torchao_fwd_bwd_ms",
        "syncless_fwd_bwd_ms",
        "torchao_fb_speedup",
        "syncless_fb_speedup",
        "ref_MFU%",
        "torchao_MFU%",
        "syncless_MFU%",
        "ref_mem_MiB",
        "torchao_mem_MiB",
        "syncless_mem_MiB",
    ]
    rows = []
    for exp in experiments:
        c, r = exp.config, exp.result
        torchao_fb_speedup = (
            r.ref_fwd_bwd_ms / r.torchao_mxfp8_fwd_bwd_ms
            if r.torchao_mxfp8_fwd_bwd_ms > 0
            else float("inf")
        )
        syncless_fb_speedup = (
            r.ref_fwd_bwd_ms / r.syncless_fwd_bwd_ms
            if r.syncless_fwd_bwd_ms > 0
            else float("inf")
        )
        rows.append(
            [
                f"({c.batch_size}, {c.seq_len}, {c.dim})",
                c.num_experts,
                dist.get_world_size(),
                f"{r.ref_fwd_bwd_ms:.2f}",
                f"{r.torchao_mxfp8_fwd_bwd_ms:.2f}",
                f"{r.syncless_fwd_bwd_ms:.2f}",
                f"{torchao_fb_speedup:.2f}x",
                f"{syncless_fb_speedup:.2f}x",
                f"{r.ref_mfu_pct:.1f}",
                f"{r.torchao_mxfp8_mfu_pct:.1f}",
                f"{r.syncless_mfu_pct:.1f}",
                f"{r.ref_peak_mem_mib:.0f}",
                f"{r.torchao_mxfp8_peak_mem_mib:.0f}",
                f"{r.syncless_peak_mem_mib:.0f}",
            ]
        )
    print("\n" + "=" * 120)
    print("MoE END-TO-END BENCHMARK: BF16 EP vs TorchAO MXFP8 EP vs Syncless MXFP8 EP")
    print("=" * 120)
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
