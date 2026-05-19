# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 tensor-parallel linear helpers for sequence-parallel training.

Implements the column-parallel protocol described in:
  cutile/rht/docs/nvfp4_column_parallel_linear.md
and the row-parallel protocol described in:
  cutile/rht/docs/nvfp4_row_parallel_linear.md

The all-gather and reduce-scatter collectives are handled inside the autograd
functions so the module forward signature stays identical to a plain nn.Linear.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_tensor,
    all_reduce,
    reduce_scatter_tensor,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax
from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)
from torchao.prototype.mx_formats.nvfp4_linear import _triton_weight_quantize_2d
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

# Column-parallel wgrad gathers RHT-transformed x shards across ranks. All ranks
# must use the same RHT basis so gathered x_col and local dy_col are compatible.
_TP_RHT_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)

_TP_STYLE_COLWISE = "colwise"
_TP_STYLE_ROWWISE = "rowwise"


def swap_first_dims(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """Fix interleave produced by NCCL all_gather on dim-0 of a colwise tensor.

    NCCL all_gather gathers along dim-0, producing rank-interleaved rows when
    the varying dimension is dim-1.  This reshape+transpose+reshape corrects
    the layout:

      Input:  [world_size * D0, D1, *rest]   (NCCL output, rows interleaved)
      Output: [D0, world_size * D1, *rest]   (correct colwise layout)

    Works for both 2-D FP4 codes [k*W, m/w//2] and 4-D swizzled scales
    [k//128*W, m/w//64, 32, 16].
    """
    D0_total = x.shape[0]
    D1 = x.shape[1]
    rest = x.shape[2:]
    D0 = D0_total // world_size
    return (
        x.reshape(world_size, D0, D1, *rest)
        .transpose(0, 1)
        .contiguous()
        .reshape(D0, world_size * D1, *rest)
    )


def _all_gather_nvfp4_rowwise(
    codes: torch.Tensor,
    sf: torch.Tensor,
    world_size: int,
    group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-gather rowwise FP4 shard along dim-0 (sequence dimension).

    Args:
        codes: [m/w, k//2] uint8 packed FP4 codes.
        sf:    [m/w//128, k//64, 32, 16] float8_e4m3fn swizzled scales.
        world_size: TP world size.
        group: ProcessGroup for the collective.

    Returns:
        (codes_full [m, k//2], sf_full [m//128, k//64, 32, 16])
    """
    codes_full = all_gather_tensor(codes, gather_dim=0, group=group)
    if isinstance(codes_full, AsyncCollectiveTensor):
        codes_full = codes_full.wait()

    # float8_e4m3fn not supported by NCCL — reinterpret as uint8 round-trip
    sf_u8 = sf.view(torch.uint8)
    sf_full_u8 = all_gather_tensor(sf_u8, gather_dim=0, group=group)
    if isinstance(sf_full_u8, AsyncCollectiveTensor):
        sf_full_u8 = sf_full_u8.wait()
    sf_full = sf_full_u8.view(torch.float8_e4m3fn)

    return codes_full, sf_full


def _async_all_gather_nvfp4_colwise(
    codes: torch.Tensor,
    sf: torch.Tensor,
    group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch async all-gather of colwise FP4 shard.

    The colwise tensor is stored as [k, m/w//2] with k on dim-0, but the
    varying dimension across ranks is m/w (dim-1).  NCCL gathers along dim-0,
    producing a [k*W, m/w//2] interleaved result.  The caller must apply
    swap_first_dims after waiting to get the correct [k, m//2] layout.

    Args:
        codes: [k, m/w//2] uint8 packed FP4 codes.
        sf:    [k//128, m/w//64, 32, 16] float8_e4m3fn swizzled scales.
        group: ProcessGroup for the collective.

    Returns:
        (codes_async, sf_async) — may be AsyncCollectiveTensors; caller must
        wait() and apply swap_first_dims before use.
    """
    codes_async = all_gather_tensor(codes, gather_dim=0, group=group)

    sf_u8 = sf.view(torch.uint8)
    sf_async_u8 = all_gather_tensor(sf_u8, gather_dim=0, group=group)

    return codes_async, sf_async_u8


@torch._dynamo.allow_in_graph
class nvfp4_col_parallel_mm(torch.autograd.Function):
    """NVFP4 column-parallel quantized matmul for sequence-parallel TP training.

    Implements the forward/backward protocol from:
      cutile/rht/docs/nvfp4_column_parallel_linear.md

    Forward:
      - Quantize x[m/w, k]: rowwise for GEMM, colwise-RHT saved for backward.
      - All-reduce amaxes across TP group for consistent scaling.
      - All-gather rowwise quantized x to [m, k//2].
      - GEMM: gathered_x @ w[n/w, k]^T = output [m, n/w].

    Backward:
      - Quantize dy[m, n/w] (no amax all-reduce; each rank has different dy).
      - Launch async all-gather of saved colwise x shard.
      - dgrad GEMM: qdy_row @ Wt^T = dx_hat [m, k].
      - Reduce-scatter dx_hat → dx [m/w, k].
      - Wait for async all-gather + swap_first_dims → qx_col_full [k, m//2].
      - wgrad GEMM: qdy_col @ qx_col_full^T = dw [n/w, k].
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
        sr_seed: torch.Tensor,
        tp_group,
        world_size: int,
    ) -> torch.Tensor:
        M_local = x.shape[0]

        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)

        # --- Amax computation + global sync ---
        col_amax, row_amax = triton_rht_amax(x, sign_vector=list(_TP_RHT_SIGN_VECTOR))
        col_amax = all_reduce(col_amax, "MAX", tp_group)
        row_amax = all_reduce(row_amax, "MAX", tp_group)
        if isinstance(col_amax, AsyncCollectiveTensor):
            col_amax = col_amax.wait()
        if isinstance(row_amax, AsyncCollectiveTensor):
            row_amax = row_amax.wait()

        # --- Quantize x with global amaxes ---
        (
            qx_col_codes,
            qx_col_sf,
            qx_row_codes,
            qx_row_sf,
        ) = triton_rht_quantize_row_col(
            x,
            stochastic_rounding=False,
            sign_vector=list(_TP_RHT_SIGN_VECTOR),
            col_global_amax=col_amax,
            row_global_amax=row_amax,
        )

        # --- 2D weight quantization ---
        # Local amax is sufficient for weights since they are not communicated.
        # Each rank only multiplies with its local shard.
        (
            W_fp4_x2,
            W_bs,
            W_gs,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
        ) = _triton_weight_quantize_2d(w)

        # --- All-gather rowwise quantized x along sequence dim ---
        x_gs = per_tensor_amax_to_scale(row_amax)
        qx_row_full, qx_row_sf_full = _all_gather_nvfp4_rowwise(
            qx_row_codes, qx_row_sf, world_size, tp_group
        )

        # --- Forward GEMM: gathered_x @ w^T = output [m, n/w] ---
        output = torch.nn.functional.scaled_mm(
            qx_row_full.view(torch.float4_e2m1fn_x2),
            W_fp4_x2.t(),
            scale_a=[qx_row_sf_full.flatten(), x_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[W_bs, W_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(
            qx_col_codes,
            qx_col_sf,
            col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        )
        ctx.tp_group = tp_group
        ctx.world_size = world_size
        ctx.has_bias = bias is not None
        ctx.local_M = M_local
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            qx_col_codes,
            qx_col_sf,
            x_col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        ) = ctx.saved_tensors
        tp_group = ctx.tp_group
        world_size = ctx.world_size

        grad_output = grad_output.contiguous()
        dev = grad_output.device

        # Independent SR offsets for the two backward quantizations
        offset_row = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )
        offset_col = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )

        # --- Quantize dy (no amax all-reduce; each rank has a different dy shard) ---
        dy_col_amax, dy_row_amax = triton_rht_amax(
            grad_output, sign_vector=list(_TP_RHT_SIGN_VECTOR)
        )
        (
            qdy_col_codes,
            qdy_col_sf,
            qdy_row_codes,
            qdy_row_sf,
        ) = triton_rht_quantize_row_col(
            grad_output,
            stochastic_rounding=True,
            sign_vector=list(_TP_RHT_SIGN_VECTOR),
            col_seed_base=sr_seed,
            col_offset_base=offset_col,
            row_offset_base=offset_row,
            row_seed_base=sr_seed ^ 1,
            col_global_amax=dy_col_amax,
            row_global_amax=dy_row_amax,
        )

        # --- Launch async all-gather of saved colwise x shard [k, m/w//2] ---
        qx_col_async, qx_col_sf_async_u8 = _async_all_gather_nvfp4_colwise(
            qx_col_codes, qx_col_sf, tp_group
        )

        # --- dgrad GEMM: qdy_row [m, n/w] @ Wt [k, n/w]^T → dx_hat [m, k] ---
        Wt_bs = Wt_sf.flatten()
        Wt_gs = per_tensor_amax_to_scale(W_amax)
        dy_row_gs = per_tensor_amax_to_scale(dy_row_amax)
        dx_hat = torch.nn.functional.scaled_mm(
            qdy_row_codes.view(torch.float4_e2m1fn_x2),
            Wt_fp4_x2.t(),
            scale_a=[qdy_row_sf.flatten(), dy_row_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[Wt_bs, Wt_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        # --- Reduce-scatter dgrad along sequence dim → dx [m/w, k] ---
        # TODO Perhaps there should be an option for fp32
        # accumulation.
        dx = reduce_scatter_tensor(dx_hat, "SUM", scatter_dim=0, group=tp_group)
        if isinstance(dx, AsyncCollectiveTensor):
            dx = dx.wait()

        # --- Wait for async all-gather + fix interleave ---
        if isinstance(qx_col_async, AsyncCollectiveTensor):
            qx_col_async = qx_col_async.wait()
        if isinstance(qx_col_sf_async_u8, AsyncCollectiveTensor):
            qx_col_sf_async_u8 = qx_col_sf_async_u8.wait()

        qx_col_full = swap_first_dims(qx_col_async, world_size)  # [k, m//2]
        qx_col_sf_full_u8 = swap_first_dims(qx_col_sf_async_u8, world_size)
        qx_col_sf_full = qx_col_sf_full_u8.view(
            torch.float8_e4m3fn
        )  # [k//128, m//64, 32, 16]

        # --- wgrad GEMM: qdy_col [n/w, m//2] @ qx_col_full [k, m//2]^T → dw [n/w, k] ---
        dy_col_gs = per_tensor_amax_to_scale(dy_col_amax)
        x_col_gs = per_tensor_amax_to_scale(x_col_amax)
        dw = torch.nn.functional.scaled_mm(
            qdy_col_codes.view(torch.float4_e2m1fn_x2),
            qx_col_full.view(torch.float4_e2m1fn_x2).t(),
            scale_a=[qdy_col_sf.flatten(), dy_col_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[qx_col_sf_full.flatten(), x_col_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        grad_bias = grad_output.sum(dim=0) if ctx.has_bias else None
        # Nones for: bias, sr_seed, tp_group, world_size
        return dx, dw, grad_bias, None, None, None


def nvfp4_col_parallel_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    sr_seed: Optional[torch.Tensor] = None,
    tp_group=None,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    """Convenience wrapper around nvfp4_col_parallel_mm.

    Args:
        x: Input [m/w, k] bfloat16.
        w: Weight shard [n/w, k] bfloat16.
        bias: Optional bias [n/w].
        sr_seed: Fixed int64 seed tensor (size=(1,)) for SR Philox key.
        tp_group: ProcessGroup for TP collectives.
        world_size: TP world size (inferred from group if None).
    """
    if tp_group is None:
        raise ValueError("tp_group is required for nvfp4_col_parallel_linear")
    if world_size is None:
        world_size = dist.get_world_size(tp_group)
    if sr_seed is None:
        sr_seed = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=x.device
        )
    return nvfp4_col_parallel_mm.apply(x, w, bias, sr_seed, tp_group, world_size)


@torch._dynamo.allow_in_graph
class nvfp4_row_parallel_mm(torch.autograd.Function):
    """NVFP4 row-parallel quantized matmul for sequence-parallel TP training.

    Implements the protocol from:
      cutile/rht/docs/nvfp4_row_parallel_linear.md

    The existing utilities in this module are enough for the implementation:
      - ``triton_rht_amax`` and ``triton_rht_quantize_row_col`` for dual-layout
        rowwise / columnwise-RHT input and gradient quantization.
      - ``_triton_weight_quantize_2d`` for rowwise and columnwise weight layouts.
      - ``_all_gather_nvfp4_rowwise`` for rowwise dy all-gather.
      - ``_async_all_gather_nvfp4_colwise`` plus ``swap_first_dims`` for colwise
        dy all-gather.
      - ``reduce_scatter_tensor`` for sequence-sharded forward output.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
        sr_seed: torch.Tensor,
        tp_group,
        world_size: int,
    ) -> torch.Tensor:
        M_local = x.shape[0]

        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)

        # --- Amax computation ---
        # For the reduce-scatter gemm pattern, calculating the true global amax using
        # all-reduce isn't necessary. The fp32 global amax is a nvfp4 quantization
        # scaling factor. Each rank computes a partial outer-product using its local
        # amax. The true output is accumulated in bf16 using reduce-scatter. For the
        # all-gather gemm pattern, each rank get entire tensor, so it must be quantized
        # with global amax using all-reduce.
        col_amax, row_amax = triton_rht_amax(x, sign_vector=list(_TP_RHT_SIGN_VECTOR))

        # --- Quantize x with local amax ---
        (
            qx_col_codes,
            qx_col_sf,
            qx_row_codes,
            qx_row_sf,
        ) = triton_rht_quantize_row_col(
            x,
            stochastic_rounding=False,
            sign_vector=list(_TP_RHT_SIGN_VECTOR),
            col_global_amax=col_amax,
            row_global_amax=row_amax,
        )

        # --- 2D weight quantization ---
        # Local amax is sufficient for weights since they are not communicated.
        # Each rank only multiplies with its local shard.
        (
            W_fp4_x2,
            W_bs,
            W_gs,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
        ) = _triton_weight_quantize_2d(w)

        # --- Forward GEMM: x @ w^T = outer product output [m, n] ---
        x_gs = per_tensor_amax_to_scale(row_amax)
        output_hat = torch.nn.functional.scaled_mm(
            qx_row_codes.view(torch.float4_e2m1fn_x2),
            W_fp4_x2.t(),
            scale_a=[qx_row_sf.flatten(), x_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[W_bs, W_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        # --- Reduce-scatter outer product output along sequence dim
        # → output [m/w, n] ---
        # TODO Perhaps there should be an option for fp32
        # accumulation.
        output = reduce_scatter_tensor(output_hat, "SUM", scatter_dim=0, group=tp_group)

        if bias is not None:
            output = output + bias

        # Save qx_t_rht, weight transpose quantization, amaxes, and sr_seed needed by
        # backward.
        ctx.save_for_backward(
            qx_col_codes,
            qx_col_sf,
            col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        )
        ctx.tp_group = tp_group
        ctx.world_size = world_size
        ctx.has_bias = bias is not None
        ctx.local_M = M_local
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            qx_col_codes,
            qx_col_sf,
            x_col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        ) = ctx.saved_tensors
        tp_group = ctx.tp_group
        world_size = ctx.world_size

        grad_output = grad_output.contiguous()
        dev = grad_output.device

        # Independent SR offsets for the two backward quantizations
        offset_row = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )
        offset_col = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )

        # --- Amax dy computation + global sync ---
        dy_col_amax, dy_row_amax = triton_rht_amax(
            grad_output, sign_vector=list(_TP_RHT_SIGN_VECTOR)
        )
        dy_col_amax = all_reduce(dy_col_amax, "MAX", tp_group)
        dy_row_amax = all_reduce(dy_row_amax, "MAX", tp_group)
        if isinstance(dy_col_amax, AsyncCollectiveTensor):
            dy_col_amax = dy_col_amax.wait()
        if isinstance(dy_row_amax, AsyncCollectiveTensor):
            dy_row_amax = dy_row_amax.wait()

        # --- Quantize dy  ---
        (
            qdy_col_codes,
            qdy_col_sf,
            qdy_row_codes,
            qdy_row_sf,
        ) = triton_rht_quantize_row_col(
            grad_output,
            stochastic_rounding=True,
            sign_vector=list(_TP_RHT_SIGN_VECTOR),
            col_seed_base=sr_seed,
            col_offset_base=offset_col,
            row_offset_base=offset_row,
            row_seed_base=sr_seed ^ 1,
            col_global_amax=dy_col_amax,
            row_global_amax=dy_row_amax,
        )

        qdy_row_full, qdy_row_sf_full = _all_gather_nvfp4_rowwise(
            qdy_row_codes, qdy_row_sf, world_size, tp_group
        )

        # --- Launch async all-gather of colwise dy shard [n, m/w//2] ---
        qdy_col_async, qdy_col_sf_async_u8 = _async_all_gather_nvfp4_colwise(
            qdy_col_codes, qdy_col_sf, tp_group
        )

        # --- dgrad GEMM: qdy_row [m, n] @ Wt [k/w, n]^T → dx [m, k/w] ---
        Wt_bs = Wt_sf.flatten()
        Wt_gs = per_tensor_amax_to_scale(W_amax)
        dy_row_gs = per_tensor_amax_to_scale(dy_row_amax)
        dx = torch.nn.functional.scaled_mm(
            qdy_row_full.view(torch.float4_e2m1fn_x2),
            Wt_fp4_x2.t(),
            scale_a=[qdy_row_sf_full.flatten(), dy_row_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[Wt_bs, Wt_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        # --- Wait for async all-gather + fix interleave ---
        if isinstance(qdy_col_async, AsyncCollectiveTensor):
            qdy_col_async = qdy_col_async.wait()
        if isinstance(qdy_col_sf_async_u8, AsyncCollectiveTensor):
            qdy_col_sf_async_u8 = qdy_col_sf_async_u8.wait()

        qdy_col_full = swap_first_dims(qdy_col_async, world_size)  # [n, m//2]
        qdy_col_sf_full_u8 = swap_first_dims(qdy_col_sf_async_u8, world_size)
        qdy_col_sf_full = qdy_col_sf_full_u8.view(
            torch.float8_e4m3fn
        )  # [n//128, m//64, 32, 16]

        # --- wgrad GEMM: qdy_col_full [n, m//2] @ qx_col [k/w, m//2]^T → dw [n, k/w] ---
        dy_col_gs = per_tensor_amax_to_scale(dy_col_amax)
        x_col_gs = per_tensor_amax_to_scale(x_col_amax)
        dw = torch.nn.functional.scaled_mm(
            qdy_col_full.view(torch.float4_e2m1fn_x2),
            qx_col_codes.view(torch.float4_e2m1fn_x2).t(),
            scale_a=[qdy_col_sf_full.flatten(), dy_col_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[qx_col_sf.flatten(), x_col_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        # dy[m/w, n] -> dy[m, n].sum(dim=0) for bias
        if ctx.has_bias:
            grad_bias_local = grad_output.sum(dim=0, keepdim=True)
            grad_bias = all_reduce(grad_bias_local, "SUM", tp_group)
        else:
            grad_bias = None

        # Nones for: bias, sr_seed, tp_group, world_size
        return dx, dw, grad_bias, None, None, None


def nvfp4_row_parallel_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    sr_seed: Optional[torch.Tensor] = None,
    tp_group=None,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    """Convenience wrapper around nvfp4_row_parallel_mm.

    Args:
        x: Input [m, k/w] bfloat16.
        w: Weight shard [n, k/w] bfloat16.
        bias: Optional replicated bias [n].
        sr_seed: Fixed int64 seed tensor (size=(1,)) for SR Philox key.
        tp_group: ProcessGroup for TP collectives.
        world_size: TP world size (inferred from group if None).
    """
    if tp_group is None:
        raise ValueError("tp_group is required for nvfp4_row_parallel_linear")
    if world_size is None:
        world_size = dist.get_world_size(tp_group)
    if sr_seed is None:
        sr_seed = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=x.device
        )
    return nvfp4_row_parallel_mm.apply(x, w, bias, sr_seed, tp_group, world_size)


class NVFP4ColwiseParallel(ColwiseParallel):
    """ColwiseParallel for NVFP4Linear with column-parallel TP.

    Subclasses ColwiseParallel to inject the process group into
    NVFP4Linear so its forward dispatches to nvfp4_col_parallel_linear.
    Weight sharding (DTensor Shard(0)) is handled by the parent class.

    The NVFP4 collectives (amax all-reduce, FP4 all-gather, reduce-scatter) run
    inside nvfp4_col_parallel_mm, so runtime hooks pass local tensor shards
    through instead of redistributing BF16 activations.
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        return (
            input_tensor.to_local()
            if isinstance(input_tensor, DTensor)
            else input_tensor
        )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor):
            if outputs.placements != output_layouts:
                outputs = outputs.redistribute(placements=output_layouts, async_op=True)
            return outputs.to_local() if use_local_output else outputs
        if use_local_output:
            return outputs
        return DTensor.from_local(outputs, device_mesh, output_layouts, run_check=False)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torchao.prototype.mx_formats.nvfp4_training import NVFP4Linear

        if not isinstance(module, NVFP4Linear):
            raise ValueError(
                f"NVFP4ColwiseParallel requires NVFP4Linear, got {type(module)}"
            )
        module.process_group = device_mesh.get_group()
        module.world_size = device_mesh.size()
        module.tensor_parallel_style = _TP_STYLE_COLWISE
        if module.weight.device.type != "meta":
            module._ensure_sr_seed(module.weight.device)
        return super()._apply(module, device_mesh)


class NVFP4RowwiseParallel(RowwiseParallel):
    """RowwiseParallel for NVFP4Linear with row-parallel TP.

    Subclasses RowwiseParallel to inject the process group into
    NVFP4Linear so its forward dispatches to nvfp4_row_parallel_linear.
    Weight sharding (DTensor Shard(1)) and replicated bias placement are handled
    by the parent class.

    The NVFP4 collectives (amax all-reduce, FP4 all-gather, reduce-scatter) run
    inside nvfp4_row_parallel_mm, so runtime hooks pass local tensor shards
    through instead of redistributing BF16 activations.
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        return (
            input_tensor.to_local()
            if isinstance(input_tensor, DTensor)
            else input_tensor
        )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor):
            if outputs.placements != output_layouts:
                outputs = outputs.redistribute(placements=output_layouts, async_op=True)
            return outputs.to_local() if use_local_output else outputs
        if use_local_output:
            return outputs
        return DTensor.from_local(outputs, device_mesh, output_layouts, run_check=False)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torchao.prototype.mx_formats.nvfp4_training import NVFP4Linear

        if not isinstance(module, NVFP4Linear):
            raise ValueError(
                f"NVFP4RowwiseParallel requires NVFP4Linear, got {type(module)}"
            )
        module.process_group = device_mesh.get_group()
        module.world_size = device_mesh.size()
        module.tensor_parallel_style = _TP_STYLE_ROWWISE
        if module.weight.device.type != "meta":
            module._ensure_sr_seed(module.weight.device)
        return super()._apply(module, device_mesh)
