# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel C: grouped MXFP8 weight-gradient GEMM (``mxfp8_grouped_gemm_wgrad``).

One Definition covers both call sites -- FC1 (``N = 2F``, ``K = D``) and FC2
(``N = D``, ``K = F``) -- with no mode flag; the shapes come from the tensors.

Per expert ``g`` over rows ``[offsets[g-1], offsets[g])``::

    dw[g] = dequant(dy_col[rows]).T @ dequant(x_col[rows])

FP32 accumulation, BF16 output. Neither transpose is materialized: both operands
arrive logically ``[R, N]`` / ``[R, K]`` with stride ``(1, R)``, which *is* a
K-contiguous row-major ``[N, R]`` / ``[K, R]``, so the free transpose is a
restride on the host and the ragged axis lands on the GEMM's contraction. The
expert is then an integer K-tile index base rather than a per-expert TMA
descriptor, exact because every per-expert row count is a multiple of
``cta_tile_k``.

Two things about this kernel that are easy to get wrong:

*Both scale buffers are WHOLE-MATRIX* ``to_blocked``, not torchao's per-group
K-groups form (``triton_mx_block_rearrange_2d_K_groups``). The two orderings
differ whenever ``N > 128`` -- whole-matrix orders blocked tiles by
``row_block * ncb_total + col_block``, per-group by ``row_block * ncb_g +
col_block`` within each group -- so feeding a per-group buffer here produces a
*block-permuted* ``dw``, which is large and structured and reads like a GEMM bug
rather than like a layout bug. The producers of these buffers are kernels A and
B in this same family, which emit the whole-matrix form.

*The epilogue never predicates its store.* A zero-token expert arrives with
``k_cnt == 0``, the core hands the epilogue a zeroed register fragment, and the
unmodified store path writes the all-zero ``dw[g]`` the contract requires. The
grid enumerates every ``(tile_m, tile_n, expert)``, so every element of ``dw`` is
written on every call with no memset. There is also no gmem input load in this
epilogue, so the ``k_cnt``-gated-load half of the tail rule has nothing to cover
here.

No output TMA and no epilogue shared memory: one row per thread means the 32
FP32 accumulators of a subtile are 32 contiguous BF16 values == 64 naturally
aligned contiguous bytes of ``dw``, so a direct vectorized ``STG`` is both
correct and fully sector-efficient.
"""

import functools
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from torchao.prototype.moe_training.kernels.mxfp8.grouped_gemm_config import (
    SF_VEC_SIZE,
    WGRAD_CONFIG,
)
from torchao.prototype.moe_training.kernels.mxfp8.grouped_gemm_core import (
    activation_gemm_view,
    launch_grouped_gemm,
)
from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_epilogue import (
    pack_bf16x2,
)
from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_validation import (
    validate_allocated_rows,
    validate_blocked_scales,
    validate_destination,
    validate_group_offsets,
    validate_grouped_operand,
)

__all__ = ["bf16_store_epilogue", "launch_grouped_gemm_wgrad"]

_E4M3 = torch.float8_e4m3fn
_BF16 = torch.bfloat16
# Widest vector the epilogue's store is allowed to assume. The run is 64 B and
# 64-B aligned, so this only has to be a divisor of that.
_VEC_BYTES = 16


@cute.jit
def _store_bf16_run(dst: cute.Tensor, elem_offset: Int32, words: cute.Tensor):
    """Store a run of packed ``bf16x2`` words at a BF16 element offset.

    ``dst`` is any BF16 destination; the run is reinterpreted as Int32, so the
    element offset must be even. Every caller's offset is a multiple of 32
    elements (see :func:`bf16_store_epilogue`), which also makes the address
    64-byte aligned and the copy four ``STG.128``.
    """
    cute.autovec_copy(
        words,
        cute.make_tensor(
            (
                cute.recast_ptr(dst.iterator, dtype=Int32) + (elem_offset >> Int32(1))
            ).align(_VEC_BYTES),
            cute.make_layout(cute.size(words)),
        ),
    )


@cute.jit
def bf16_store_epilogue(
    tTR_rAcc,
    tTR_cAcc_s,
    tiled_copy_t2r,
    epi_tidx,
    subtile_idx: cutlass.Constexpr,
    tile,
    epi_smem,
    out,
    cfg: cutlass.Constexpr,
):
    """Round the FP32 accumulator subtile to BF16 and store it. No quantization.

    ``out`` is ``(mDw,)`` with ``mDw`` the ``(N, K, G)`` view of the contiguous
    ``[G, N, K]`` destination.

    The register-to-column map is taken from ``tTR_cAcc``, whose column
    coordinates fold to Python ints at trace time, and the run is required to be
    contiguous and increasing. That check is what licenses both the ``pack``
    pairing and the single vectorized store: if a thread owned more than one row
    of the epilogue tile its columns would repeat instead of forming a run, so
    this also establishes the one-row-per-thread property the store address
    assumes, rather than trusting it.
    """
    num_acc = cutlass.const_expr(cute.size(tTR_rAcc))
    cols = []
    for v in cutlass.range_constexpr(num_acc):
        cols.append(tTR_cAcc_s[v][1])
    frag_col = cols[0]
    if cutlass.const_expr(
        num_acc % 2 != 0
        or not all(isinstance(c, int) for c in cols)
        or tuple(cols) != tuple(range(frag_col, frag_col + num_acc))
    ):
        raise ValueError(
            f"the wgrad epilogue needs an even, contiguous, increasing column run "
            f"per thread to pack and store BF16 pairs, but tTR_cAcc gave {cols}"
        )

    # cvt.rn.bf16x2.f32 packs the second source into the low half, so column
    # frag_col + 2j lands at the lower address -- row-major order for dw.
    words = cute.make_rmem_tensor((num_acc // 2,), Int32)
    for j in cutlass.range_constexpr(num_acc // 2):
        words[j] = pack_bf16x2(tTR_rAcc[2 * j + 1], tTR_rAcc[2 * j])

    # Address from the destination's real strides, which are static ints here,
    # rather than from its extents: the only layout property the store actually
    # needs is that the K axis is contiguous.
    gDw = out[0]
    strides = gDw.stride
    if cutlass.const_expr(strides[1] != 1):
        raise ValueError(
            f"the wgrad epilogue stores a contiguous run along K, so dw's K "
            f"stride must be 1, got layout {gDw.layout}"
        )
    # Row from tTR_cAcc, not from epi_tidx: the contiguity check above proves
    # the fragment is one row, and this is the coordinate that names it.
    row = tile.row_base + tTR_cAcc_s[0][0]
    elem = (
        row * Int32(strides[0])
        + (tile.col_base + Int32(frag_col))
        + tile.expert * Int32(strides[2])
    )
    _store_bf16_run(gDw, elem, words)


@cute.jit
def _wgrad_entry(mA, mB, sfa, sfb, offs, mDw, stream):
    """Trace entry point: dynamic tensors only, config and epilogue closed over.

    ``launch_grouped_gemm`` is a trace body, so calling it directly retraces the
    whole kernel on every launch. Everything Constexpr is bound here so that
    :func:`cute.compile` can hand back an executor that takes only these
    arguments; passing a Constexpr to that executor raises "cannot be converted
    to pointer".
    """
    launch_grouped_gemm(
        mA,
        mB,
        sfa,
        sfb,
        offs,
        (mDw,),
        stream,
        WGRAD_CONFIG,
        bf16_store_epilogue,
    )


@functools.cache
def _wgrad_executor_slot(key: Tuple) -> list:
    """One memo slot per compiled shape.

    The executor cannot be built from ``key`` alone: the shared core needs static
    shapes (the blocked scale-factor layout and the grid are both built from
    them), so a symbolic ``cute.sym_int`` compile is not available and the first
    real call's tensors are what gets compiled. ``functools.cache`` therefore
    keys the slot and the caller fills it once.
    """
    return []


def _validate_wgrad_operands(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets, dw):
    """Host-only precondition check. Metadata and pointers, never offset values.

    The custom op validates its own inputs, but this launcher is also reachable
    directly from the DSL and it owns the destination, which the op does not
    check. Every gate here is metadata-derivable, so it costs no synchronization
    and stays traceable.
    """
    if dy_col_q.ndim != 2 or x_col_q.ndim != 2:
        raise ValueError(
            "dy_col_q and x_col_q must be 2D logical [R, N] and [R, K], got "
            f"{tuple(dy_col_q.shape)} and {tuple(x_col_q.shape)}"
        )
    rows, out_features = dy_col_q.shape
    x_rows, in_features = x_col_q.shape
    if x_rows != rows:
        raise ValueError(
            f"dy_col_q and x_col_q must share the row dim: {rows} vs {x_rows}"
        )
    groups = offsets.numel()
    device = dy_col_q.device

    validate_allocated_rows(rows)
    # N and K are the GEMM's two free axes; both are tiled with no tail path, so
    # reject a non-multiple here rather than from inside the trace.
    for name, value, tile in (
        ("dy_col_q's N", out_features, WGRAD_CONFIG.cta_tile_m),
        ("x_col_q's K", in_features, WGRAD_CONFIG.cta_tile_n),
    ):
        if value <= 0 or value % tile != 0:
            raise ValueError(
                f"{name} must be a positive multiple of {tile}, got {value}"
            )
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    # Column-major, so the transposes below are free restrides.
    validate_grouped_operand(
        dy_col_q,
        name="dy_col_q",
        shape=(rows, out_features),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        x_col_q,
        name="x_col_q",
        shape=(rows, in_features),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        dy_col_sf,
        name="dy_col_sf",
        logical_rows=out_features,
        logical_cols=rows // SF_VEC_SIZE,
        device=device,
    )
    validate_blocked_scales(
        x_col_sf,
        name="x_col_sf",
        logical_rows=in_features,
        logical_cols=rows // SF_VEC_SIZE,
        device=device,
    )
    # validate_blocked_scales checks dtype, length and device but not the
    # pointer, and both scale buffers are TMA operands: a contiguous view with a
    # storage offset can be 2-byte aligned. Only the launcher promises the TMA
    # alignment, so only the launcher can require it.
    for name, buf in (("dy_col_sf", dy_col_sf), ("x_col_sf", x_col_sf)):
        if buf.data_ptr() % 32 != 0:
            raise ValueError(
                f"{name} must be 32-byte aligned for its TMA descriptor, but its "
                f"data pointer is {buf.data_ptr() % 32} bytes past an aligned "
                "address"
            )
    validate_destination(
        dw,
        name="dw_bf16",
        shape=(groups, out_features, in_features),
        stride=(out_features * in_features, in_features, 1),
        dtype=_BF16,
        device=device,
    )
    # The epilogue computes its destination element index in Int32.
    if groups * out_features * in_features >= 2**31:
        raise ValueError(
            f"dw_bf16 has {groups * out_features * in_features} elements, which "
            "does not fit the epilogue's int32 element index; the store address "
            "arithmetic would wrap"
        )
    return rows, out_features, in_features, groups


def launch_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets, dw):
    """Grouped MXFP8 wgrad into a caller-allocated BF16 ``[G, N, K]`` destination.

    Inputs are the columnwise-quantized outputs of kernels A and B (or of the
    standalone 32x1 quantizer): ``dy_col_q`` E4M3 logical ``[R, N]`` stride
    ``(1, R)`` with ``dy_col_sf`` blocked for logical ``[N, R/32]``, and
    ``x_col_q`` / ``x_col_sf`` likewise for ``[R, K]``. ``offsets`` is the int32
    CUDA ``[G]`` vector of exclusive group ends and is never read on the host.

    Every element of ``dw`` is written, including the all-zero slice of a
    zero-token expert.
    """
    import cuda.bindings.driver as cuda

    rows, out_features, in_features, groups = _validate_wgrad_operands(
        dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets, dw
    )
    if rows == 0:
        # Every expert has zero rows, so every slice is the zero matrix. Unlike
        # the other two kernels the destination is NOT empty here, and the
        # contraction is, so this cannot be expressed as a launch.
        dw.zero_()
        return

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    args = (
        # The free transpose: logical [R, N] stride (1, R) IS a K-contiguous
        # [N, R], so both operands become ordinary K-major GEMM operands and the
        # ragged axis becomes the contraction.
        from_dlpack(activation_gemm_view(dy_col_q.t()), assumed_align=16),
        from_dlpack(activation_gemm_view(x_col_q.t()), assumed_align=16),
        # Carried flat and recast to E8M0 inside the trace; E8M0 has no DLPack
        # dtype, so hand over the raw bytes.
        from_dlpack(dy_col_sf.view(torch.uint8), assumed_align=16),
        from_dlpack(x_col_sf.view(torch.uint8), assumed_align=16),
        from_dlpack(offsets, assumed_align=4),
        # (G, N, K) contiguous -> (N, K, G): the expert is the L coordinate.
        from_dlpack(dw.permute(1, 2, 0), assumed_align=16),
        stream,
    )

    slot = _wgrad_executor_slot((rows, out_features, in_features, groups))
    if not slot:
        slot.append(cute.compile(_wgrad_entry, *args))
    slot[0](*args)
