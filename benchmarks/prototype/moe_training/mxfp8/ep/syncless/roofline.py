# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass

from tabulate import tabulate

from torchao.testing.training.roofline_utils import BYTES_PER_EL_BF16, get_specs


@dataclass(frozen=True)
class RooflineConfig:
    batch_size: int
    seq_len: int
    dim: int
    hidden_dim: int
    num_experts: int
    top_k: int


@dataclass(frozen=True)
class RooflineBreakdown:
    total_ms: float
    gemm_ms: float
    quant_ms: float
    ep_motion_ms: float
    token_padding_ms: float
    activation_ms: float


@dataclass(frozen=True)
class RooflineComparison:
    config: RooflineConfig
    ref: RooflineBreakdown
    torchao: RooflineBreakdown
    syncless: RooflineBreakdown


DEFAULT_CONFIG = RooflineConfig(
    batch_size=1,
    seq_len=4096,
    dim=7168,
    hidden_dim=2048,
    num_experts=16,
    top_k=8,
)

DEFAULT_NVLINK_BW_BYTES_SEC = 900e9
DEFAULT_NVLINK_ACHIEVABLE_PCT = 0.80


def _roofline_ms_from_flops(flops: float, peak_flops_s: float) -> float:
    return flops / peak_flops_s * 1e3


def _roofline_ms_from_bytes(num_bytes: float, peak_bytes_s: float) -> float:
    return num_bytes / peak_bytes_s * 1e3


def _effective_bf16_peak(specs) -> float:
    return specs["bf16_peak_tops"] * specs.get("pct_achievable_gemm_tops", 1.0)


def _effective_fp8_peak(specs) -> float:
    return specs["fp8_peak_tops"] * specs.get("pct_achievable_gemm_tops", 1.0)


def _effective_mem_bw(specs) -> float:
    return specs["peak_mem_bw_bytes_sec"] * specs.get("pct_achievable_mem_bw", 1.0)


def _mxfp8_quant_bytes_2d(rows: int, cols: int, block_size: int = 32) -> int:
    # bf16 input read + fp8 data write + e8m0 scale write.
    return rows * cols * BYTES_PER_EL_BF16 + rows * cols + rows * (cols // block_size)


def _mxfp8_quant_bytes_3d(
    groups: int, rows: int, cols: int, block_size: int = 32
) -> int:
    # bf16 input read + fp8 data write + e8m0 scale write.
    return (
        groups * rows * cols * BYTES_PER_EL_BF16
        + groups * rows * cols
        + groups * rows * (cols // block_size)
    )


def _scale_rearrange_bytes(*shape: int) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    # uint8 scale read + blocked scale write.
    return 2 * numel


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _balanced_padded_rows(total_rows: int, groups: int, alignment: int) -> int:
    rows_per_group, remainder = divmod(total_rows, groups)
    padded_rows = 0
    for group_idx in range(groups):
        rows = rows_per_group + int(group_idx < remainder)
        padded_rows += _round_up(max(rows, alignment), alignment)
    return padded_rows


def _mxfp8_grouped_linear_fwd_bwd_roofline_ms(
    M: int,
    K: int,
    N: int,
    G: int,
    specs,
    include_wgrad: bool = True,
    wgrad_dtype: str = "fp8",
    include_scale_rearrange: bool = True,
) -> tuple[float, float]:
    """Return (gemm_ms, quant_and_scale_ms) for one grouped linear fwd+bwd.

    Shapes follow the existing MXFP8 grouped-GEMM roofline helper:
      fwd:         (M, K) @ (G, K, N) -> (M, N)
      bwd input:   (M, N) @ (G, N, K) -> (M, K)
      bwd weight:  grouped (N, M) @ (M, K) -> (G, N, K)
    """
    block_size = 32
    peak_bw = _effective_mem_bw(specs)
    fp8_peak = _effective_fp8_peak(specs)
    bf16_peak = _effective_bf16_peak(specs)

    fwd_flops = 2 * M * K * N
    dgrad_flops = 2 * M * N * K
    wgrad_flops = 2 * N * M * K

    if include_wgrad:
        if wgrad_dtype == "fp8":
            gemm_ms = _roofline_ms_from_flops(
                fwd_flops + dgrad_flops + wgrad_flops, fp8_peak
            )
        elif wgrad_dtype == "bf16":
            gemm_ms = _roofline_ms_from_flops(
                fwd_flops + dgrad_flops, fp8_peak
            ) + _roofline_ms_from_flops(wgrad_flops, bf16_peak)
        else:
            raise ValueError(f"Unsupported wgrad dtype: {wgrad_dtype}")
    else:
        gemm_ms = _roofline_ms_from_flops(fwd_flops + dgrad_flops, fp8_peak)

    # Forward quant + optional scale layout.
    fwd_quant_bytes = _mxfp8_quant_bytes_2d(
        M, K, block_size
    ) + _mxfp8_quant_bytes_3d(G, N, K, block_size)
    fwd_scale_bytes = 0
    if include_scale_rearrange:
        fwd_scale_bytes = _scale_rearrange_bytes(
            M, K // block_size
        ) + _scale_rearrange_bytes(G, N, K // block_size)

    # Backward-input quant + optional scale layout.
    dgrad_quant_bytes = _mxfp8_quant_bytes_2d(
        M, N, block_size
    ) + _mxfp8_quant_bytes_3d(G, K, N, block_size)
    dgrad_scale_bytes = 0
    if include_scale_rearrange:
        dgrad_scale_bytes = _scale_rearrange_bytes(
            M, N // block_size
        ) + _scale_rearrange_bytes(G, K, N // block_size)

    quant_bytes = (
        fwd_quant_bytes + fwd_scale_bytes + dgrad_quant_bytes + dgrad_scale_bytes
    )

    if include_wgrad and wgrad_dtype == "fp8":
        wgrad_quant_bytes = _mxfp8_quant_bytes_2d(
            N, M, block_size
        ) + _mxfp8_quant_bytes_2d(M, K, block_size)
        wgrad_scale_bytes = 0
        if include_scale_rearrange:
            wgrad_scale_bytes = _scale_rearrange_bytes(
                N, M // block_size
            ) + _scale_rearrange_bytes(K, M // block_size)
        quant_bytes += wgrad_quant_bytes + wgrad_scale_bytes

    return gemm_ms, _roofline_ms_from_bytes(quant_bytes, peak_bw)


def _activation_roofline_ms(M: int, H: int, specs) -> float:
    # Approximate SwiGLU fwd+bwd as memory traffic over bf16 activations:
    # fwd reads two H halves and writes H; bwd reads grad + saved halves and writes 2H.
    activation_bytes = 16 * M * H
    return _roofline_ms_from_bytes(activation_bytes, _effective_mem_bw(specs))


def _syncless_activation_roofline_ms(M: int, H: int, specs) -> float:
    block_size = 32
    # silu_mul_fw_mxfp8 reads h13 from the saved-activations buffer and writes
    # h directly to reusable MXFP8 expert-compute buffers.
    fwd_bytes = (
        M * (2 * H) * BYTES_PER_EL_BF16 # read h13
        + M * H # write h
        + M * (H // block_size) # write h scales
    )

    # silu_mul_bw reads h13 and grad_h, then writes bf16 h and grad_h13
    # directly to reusable expert-compute buffers for the downstream wgrads/dgrad.
    bwd_bytes = (
        M * (2 * H) * BYTES_PER_EL_BF16 # read h13
        + M * H * BYTES_PER_EL_BF16 # write h
        + M * H * BYTES_PER_EL_BF16 # write h
        + M * (2 * H) * BYTES_PER_EL_BF16 # write grad h13 
    )
    return _roofline_ms_from_bytes(fwd_bytes + bwd_bytes, _effective_mem_bw(specs))


def _torchao_token_padding_roofline_ms(
    rows: int,
    dim: int,
    num_local_experts: int,
    specs,
    alignment: int = 32,
) -> float:
    padded_rows = _balanced_padded_rows(rows, num_local_experts, alignment)
    padding_bytes = (rows + padded_rows) * dim * BYTES_PER_EL_BF16
    return _roofline_ms_from_bytes(padding_bytes, _effective_mem_bw(specs))


def _syncless_expert_quant_roofline_ms(
    M: int,
    D: int,
    H: int,
    G: int,
    specs,
) -> float:
    """Approximate non-GEMM expert quant/format traffic in syncless experts.

    This follows torchao/prototype/moe_training/ep/syncless/moe.py:
    dispatch already quantized x to MXFP8, w2 wgrad is MXFP8, and all syncless
    quantization kernels are assumed to produce blocked scales directly.
    The only separate scale-layout transform modeled here is for forward
    dispatched activation scales, which are row-major for transport.
    Writes go directly to preallocated symmetric-memory, saved-activation, or
    expert-compute buffers; no allocation/copy penalty is modeled.
    """
    block_size = 32

    # Directly save x for w13 wgrad: dequant/requant from dispatched row-major
    # 1x32 MXFP8 into the saved 32x1 col-major MXFP8 activation buffer.
    save_x_format_bytes = 2 * (M * D + M * (D // block_size))

    # Forward w13: quantize w13 directly to blocked-scale layout. The
    # dispatched activation scales are row-major for transport, so they need
    # one transform to blocked layout after arrival.
    w13_fwd_bytes = (
        _mxfp8_quant_bytes_3d(G, 2 * H, D, block_size)
        + _scale_rearrange_bytes(M, D // block_size)
    )

    # Forward w2: h is already MXFP8 with blocked scales; quantize w2 directly
    # to blocked-scale layout.
    w2_fwd_bytes = _mxfp8_quant_bytes_3d(G, D, H, block_size)

    # Backward starts by reading grad_out once and writing both row-major and
    # transposed MXFP8 forms with blocked scales.
    grad_out_quant_bytes = (
        M * D * BYTES_PER_EL_BF16
        + 2 * M * D
        + 2 * M * (D // block_size)
    )

    # w2 dgrad: grad_out is already MXFP8; mxfp8_quantize_cuda_3d produces
    # blocked w2 scales directly.
    w2_dgrad_bytes = _mxfp8_quant_bytes_3d(G, D, H, block_size)

    # w13 wgrad: quantize grad_h13 directly to blocked-scale layout. Saved x
    # is modeled as already usable by the cutedsl GEMM.
    w13_wgrad_bytes = _mxfp8_quant_bytes_2d(M, 2 * H, block_size)

    # w2 wgrad: h is materialized as bf16 by silu_mul_bw, then quantized to
    # MXFP8 for cutedsl_grouped_gemm. grad_out_t was produced above.
    w2_wgrad_bytes = _mxfp8_quant_bytes_2d(M, H, block_size)

    # w13 dgrad currently calls _compute_dgrad on bf16 grad_h13, so it
    # requantizes grad_h13 and w13 separately, directly to blocked-scale layout.
    w13_dgrad_bytes = (
        _mxfp8_quant_bytes_2d(M, 2 * H, block_size)
        + _mxfp8_quant_bytes_3d(G, D, 2 * H, block_size)
    )

    total_bytes = (
        save_x_format_bytes
        + w13_fwd_bytes
        + w2_fwd_bytes
        + grad_out_quant_bytes
        + w2_dgrad_bytes
        + w13_wgrad_bytes
        + w2_wgrad_bytes
        + w13_dgrad_bytes
    )
    return _roofline_ms_from_bytes(total_bytes, _effective_mem_bw(specs))


def compute_e2e_roofline(
    config: RooflineConfig,
    world_size: int = 2,
    gpu_name: str = "NVIDIA B200",
    comm_bw_bytes_sec: float = DEFAULT_NVLINK_BW_BYTES_SEC,
    comm_achievable_pct: float = DEFAULT_NVLINK_ACHIEVABLE_PCT,
) -> RooflineComparison:
    """Peak-hardware roofline model for the three fwd+bwd EP MoE variants.

    The model intentionally ignores CPU-visible D2H synchronization latency and
    metadata collectives. Local memory traffic uses the achievable HBM bandwidth
    from roofline_utils.py; inter-rank EP payload traffic uses
    comm_bw_bytes_sec * comm_achievable_pct.
    """
    specs = get_specs(gpu_name)
    peak_bw = _effective_mem_bw(specs)
    effective_comm_bw = comm_bw_bytes_sec * comm_achievable_pct
    bf16_peak = _effective_bf16_peak(specs)
    fp8_peak = _effective_fp8_peak(specs)

    M = config.batch_size * config.seq_len * config.top_k
    D = config.dim
    H = config.hidden_dim
    if config.num_experts < world_size or config.num_experts % world_size != 0:
        raise ValueError(
            "num_experts must be divisible by world_size and at least world_size "
            f"(got num_experts={config.num_experts}, world_size={world_size})"
        )
    G_local = config.num_experts // world_size

    # Expert GEMMs: w13 has output 2H, w2 has output D.
    w13_flops = 3 * 2 * M * D * (2 * H)
    w2_flops = 3 * 2 * M * H * D
    ref_gemm_ms = _roofline_ms_from_flops(w13_flops + w2_flops, bf16_peak)
    torchao_gemm_ms = _roofline_ms_from_flops(w13_flops + w2_flops, fp8_peak)
    syncless_gemm_ms = _roofline_ms_from_flops(w13_flops + w2_flops, fp8_peak)

    w13_gemm_ms, w13_torchao_quant_ms = _mxfp8_grouped_linear_fwd_bwd_roofline_ms(
        M, D, 2 * H, G_local, specs, include_scale_rearrange=False
    )
    w2_gemm_ms, w2_torchao_quant_ms = _mxfp8_grouped_linear_fwd_bwd_roofline_ms(
        M, H, D, G_local, specs, include_scale_rearrange=False
    )
    # Keep the direct GEMM formulas above as the source of truth for mixed
    # precision; the helper's GEMM component is useful as a consistency check.
    assert abs((w13_gemm_ms + w2_gemm_ms) - torchao_gemm_ms) < 1e-9
    torchao_quant_ms = w13_torchao_quant_ms + w2_torchao_quant_ms

    syncless_quant_ms = _syncless_expert_quant_roofline_ms(
        M, D, H, G_local, specs
    )

    # Standard EP path: four bf16 all-to-all payload movements in fwd+bwd plus
    # four local permute/unpermute-style bf16 reorderings. NVLink bytes are
    # counted as transfer payload, while local reorder bytes are HBM read+write.
    bf16_payload_bytes = M * D * BYTES_PER_EL_BF16
    standard_ep_comm_bytes = 4 * bf16_payload_bytes
    standard_ep_local_bytes = 4 * 2 * bf16_payload_bytes

    # Syncless path: forward dispatch quantizes once and pushes MXFP8 data plus
    # scales; combine, combine-bwd, and dispatch-bwd push bf16 payloads. There
    # is no separate local expert-major permute.
    mxfp8_payload_bytes = M * D + M * (D // 32)
    syncless_dispatch_quant_bytes = _mxfp8_quant_bytes_2d(M, D)
    syncless_comm_bytes = mxfp8_payload_bytes + 3 * bf16_payload_bytes

    ref_ep_ms = _roofline_ms_from_bytes(
        standard_ep_comm_bytes, effective_comm_bw
    ) + _roofline_ms_from_bytes(standard_ep_local_bytes, peak_bw)
    torchao_ep_ms = ref_ep_ms
    syncless_ep_ms = _roofline_ms_from_bytes(
        syncless_comm_bytes, effective_comm_bw
    ) + _roofline_ms_from_bytes(syncless_dispatch_quant_bytes, peak_bw)
    torchao_token_padding_ms = _torchao_token_padding_roofline_ms(
        M, D, G_local, specs, alignment=32
    )

    activation_ms = _activation_roofline_ms(M, H, specs)
    syncless_activation_ms = _syncless_activation_roofline_ms(M, H, specs)

    ref = RooflineBreakdown(
        total_ms=ref_gemm_ms + ref_ep_ms + activation_ms,
        gemm_ms=ref_gemm_ms,
        quant_ms=0.0,
        ep_motion_ms=ref_ep_ms,
        token_padding_ms=0.0,
        activation_ms=activation_ms,
    )
    torchao = RooflineBreakdown(
        total_ms=(
            torchao_gemm_ms
            + torchao_quant_ms
            + torchao_ep_ms
            + torchao_token_padding_ms
            + activation_ms
        ),
        gemm_ms=torchao_gemm_ms,
        quant_ms=torchao_quant_ms,
        ep_motion_ms=torchao_ep_ms,
        token_padding_ms=torchao_token_padding_ms,
        activation_ms=activation_ms,
    )
    syncless = RooflineBreakdown(
        total_ms=(
            syncless_gemm_ms
            + syncless_quant_ms
            + syncless_ep_ms
            + syncless_activation_ms
        ),
        gemm_ms=syncless_gemm_ms,
        quant_ms=syncless_quant_ms,
        ep_motion_ms=syncless_ep_ms,
        token_padding_ms=0.0,
        activation_ms=syncless_activation_ms,
    )
    return RooflineComparison(config=config, ref=ref, torchao=torchao, syncless=syncless)


def print_roofline_results(
    world_size: int,
    gpu_name: str = "NVIDIA B200",
    batch_sizes: tuple[int, ...] = (1, 4),
    comm_bw_bytes_sec: float = DEFAULT_NVLINK_BW_BYTES_SEC,
    num_experts: int = DEFAULT_CONFIG.num_experts,
    comm_achievable_pct: float = DEFAULT_NVLINK_ACHIEVABLE_PCT,
) -> None:
    experiments = [
        compute_e2e_roofline(
            RooflineConfig(
                batch_size=batch_size,
                seq_len=DEFAULT_CONFIG.seq_len,
                dim=DEFAULT_CONFIG.dim,
                hidden_dim=DEFAULT_CONFIG.hidden_dim,
                num_experts=num_experts,
                top_k=DEFAULT_CONFIG.top_k,
            ),
            world_size=world_size,
            gpu_name=gpu_name,
            comm_bw_bytes_sec=comm_bw_bytes_sec,
            comm_achievable_pct=comm_achievable_pct,
        )
        for batch_size in batch_sizes
    ]
    rows = []
    for exp in experiments:
        c = exp.config
        rows.append(
            [
                c.batch_size,
                f"{exp.ref.total_ms:.4f}",
                f"{exp.torchao.total_ms:.4f}",
                f"{exp.syncless.total_ms:.4f}",
                f"{exp.ref.total_ms / exp.torchao.total_ms:.2f}x",
                f"{exp.ref.total_ms / exp.syncless.total_ms:.2f}x",
                f"{exp.ref.gemm_ms:.4f}",
                f"{exp.torchao.gemm_ms:.4f}",
                f"{exp.syncless.gemm_ms:.4f}",
                f"{exp.torchao.quant_ms:.4f}",
                f"{exp.syncless.quant_ms:.4f}",
                f"{exp.ref.ep_motion_ms:.4f}",
                f"{exp.torchao.token_padding_ms:.4f}",
                f"{exp.syncless.ep_motion_ms:.4f}",
            ]
        )
    print("\n" + "=" * 120)
    print(
        f"PEAK {gpu_name} ROOFLINE MODEL: "
        f"local batches {', '.join(str(b) for b in batch_sizes)}, "
        f"experts={num_experts}, "
        f"world_size={world_size}, "
        f"comm_bw={comm_bw_bytes_sec / 1e9:.0f} GB/s "
        f"@ {comm_achievable_pct * 100:.0f}%"
    )
    print("=" * 120)
    print(
        tabulate(
            rows,
            headers=[
                "local_B",
                "ref_ms",
                "torchao_ms",
                "syncless_ms",
                "torchao_vs_ref",
                "syncless_vs_ref",
                "ref_gemm",
                "torchao_gemm",
                "syncless_gemm",
                "torchao_quant",
                "syncless_quant",
                "std_ep_motion",
                "torchao_pad",
                "syncless_ep_motion",
            ],
            tablefmt="grid",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print peak roofline estimates for MXFP8 EP MoE variants."
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Expert-parallel world size. Default: 2",
    )
    parser.add_argument(
        "--gpu-name",
        default="NVIDIA B200",
        help="GPU name key from torchao.testing.training.roofline_utils. Default: NVIDIA B200",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4],
        help="Local batch sizes to model. Default: 1 4",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=DEFAULT_CONFIG.num_experts,
        help=f"Total number of experts. Default: {DEFAULT_CONFIG.num_experts}",
    )
    parser.add_argument(
        "--comm-bandwidth-gb-s",
        type=float,
        default=DEFAULT_NVLINK_BW_BYTES_SEC / 1e9,
        help="Inter-rank EP communication bandwidth in GB/s. Default: 900",
    )
    parser.add_argument(
        "--comm-achievable-pct",
        type=float,
        default=DEFAULT_NVLINK_ACHIEVABLE_PCT,
        help="Fraction of peak comm bandwidth achievable by comm kernels. Default: 0.80",
    )
    args = parser.parse_args()
    print_roofline_results(
        world_size=args.world_size,
        gpu_name=args.gpu_name,
        batch_sizes=tuple(args.batch_sizes),
        comm_bw_bytes_sec=args.comm_bandwidth_gb_s * 1e9,
        num_experts=args.experts,
        comm_achievable_pct=args.comm_achievable_pct,
    )


if __name__ == "__main__":
    main()
