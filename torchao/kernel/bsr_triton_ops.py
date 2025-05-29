# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import os
from typing import Optional

import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

if TORCH_VERSION_AT_LEAST_2_4:
    from torch._dynamo.utils import warn_once
else:
    import warnings

    warn_once = warnings.warn
from torch.sparse._triton_ops import (
    broadcast_batch_dims,
    launch_kernel,
    prepare_inputs,
    ptr_stride_extractor,
    tile_to_blocksize,
)
from torch.sparse._triton_ops_meta import get_meta, minimize, update
from torch.utils._triton import has_triton

AUTOTUNE = os.getenv("BSR_AUTOTUNE", False)

def tune_bsr_dense_addmm(
    input,
    bsr,
    dense,
    *,
    beta=1,
    alpha=1,
    left_alpha=None,
    right_alpha=None,
    out=None,
    store=False,
    verbose=False,
    force=False,
    opname=None,
):
    """Tune bsr_dense_addmm kernel parameters against the given inputs.

    When store is True, the tuning results will be stored in the
    database of kernel parameters.
    """
    import triton

    if opname is None:
        opname = "bsr_dense_addmm"

    N = dense.shape[-1]
    values = bsr.values()
    crow_indices = bsr.crow_indices()
    batch_ndim = crow_indices.dim() - 1
    M, K = bsr.shape[batch_ndim : batch_ndim + 2]
    BM, BK = values.shape[batch_ndim + 1 : batch_ndim + 3]

    # Reference parameters is a set of parameters that leads to a
    # successful kernel call and the corresponding timing is used as a
    # reference for computing speedups. Avoid changing the reference
    # parameters when possible.
    reference_meta = dict(
        GROUP_SIZE_ROW=1, num_stages=4, num_warps=4, SPLIT_N=max(N // BM, 1)
    )

    # Compute the key of parameters:
    sparsity = round(1 - bsr._nnz() * BM * BK / (M * K), 2)
    dtype = bsr.dtype
    if out is None:
        out_dtype = dtype
    else:
        out_dtype = out.dtype
    if out_dtype is dtype:
        version_dtype = dtype
    else:
        version_dtype = (dtype, out_dtype)
    version = (0, version_dtype, sparsity)
    key = (M, K, N, BM, BK, beta == 0, beta == 1, alpha == 1, N % max(N // BM, 1)== 0)

    # For tuning, for an initial state, use parameters from the
    # database if available, otherwise, use the reference parameters.
    initial_meta = get_meta(opname, key, version=version, exact=True)
    if initial_meta is None:
        may_skip_update = False
        initial_meta = get_meta(opname, key, version=(0, dtype, 0.5), exact=True)
        if initial_meta is None:
            initial_meta = reference_meta
    elif not force:
        return initial_meta
    else:
        may_skip_update = True

    # The target function that is minimized in the tuning process:
    def bench(meta, input=input, bsr=bsr, dense=dense, alpha=alpha, out=out):
        def test_func():
            return bsr_dense_addmm(
                input,
                bsr,
                dense,
                beta=beta,
                alpha=alpha,
                left_alpha=left_alpha,
                right_alpha=right_alpha,
                meta=meta,
                out=out,
            )

        return triton.testing.do_bench(test_func, warmup=500, rep=100)

    # The step function that increments a specified meta parameter:
    def step_meta_parameter(name, value, direction, meta, M=M, N=N, K=K, BM=BM, BK=BK):
        # return next value in positive or negative direction, or
        # input value if the step will result an invalid
        # value. The input value is assumed to be valid.
        is_log = name in {"SPLIT_N", "num_warps"}
        min_value = dict(SPLIT_N=1, num_warps=1, num_stages=1, GROUP_SIZE_ROW=1)[name]
        max_value = dict(SPLIT_N=max(N // BM, 1)).get(name)
        value_step = dict(SPLIT_N=2, num_warps=2, num_stages=1, GROUP_SIZE_ROW=2)[name]
        if is_log:
            next_value = (
                value * value_step**direction
                if direction > 0
                else value // (value_step ** abs(direction))
            )
        else:
            next_value = value + value_step * direction
        if min_value is not None:
            next_value = max(next_value, min_value)
        if max_value is not None:
            next_value = min(next_value, max_value)
        if name == "SPLIT_N" and N % (next_value * BM) != 0:
            return value
        return next_value

    # Tune:
    meta, speedup, timing, sensitivity_message = minimize(
        bench,
        initial_meta,
        reference_meta,
        step_meta_parameter,
        max_step=2,
        verbose=verbose,
    )
    if verbose:
        print(f"-> {sensitivity_message}, {speedup=:.1f} %, {timing=:.3f} ms")

    if store and not (
        may_skip_update and meta == initial_meta and initial_meta is not reference_meta
    ):
        device_name = torch.cuda.get_device_name()
        update(
            opname,
            device_name,
            version,
            key,
            tuple(meta[k] for k in sorted(meta)),
        )

    return meta


def bsr_dense_addmm_meta(
    M,
    K,
    N,
    # Ms,
    # Ks,
    blocksize: tuple[int,int],
    beta,
    alpha,
    SPLIT_N=None,
    GROUP_SIZE_ROW=None,
    num_warps=None,
    num_stages=None,
    sparsity=None,
    dtype=None,
    out_dtype=None,
    _version=0,
    **extra,
):
    # Specifying _version is useful for situations when one wants to
    # discard existing triton kernel tuning results, say, in testing
    # bsr_dense_addmm_meta functionality.
    if dtype is None:
        dtype = torch.float16
    if out_dtype is None:
        out_dtype = dtype
    if sparsity is None:
        sparsity = 0.5
    # if {SPLIT_N, num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
    BM, BK = blocksize
    #calculate a default SPLIT_N that ensures BN is valid
    default_split_n = max(N // BM, 1)
    if {num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = torch.cuda.get_device_name()
        key = (M, K, N, BM, BK, beta == 0, beta == 1, alpha == 1, N % default_split_n == 0)
        # If no parameters are specified, use the default parameters.
        # if AUTOTUNE:
            # If AUTOTUNE is True, try to find the optimal triton kernel
            # parameters. If the optimal triton kernel parameters are not
            # found, use the default parame,ters.
        if dtype is out_dtype:
            version_dtype = dtype
        else:
            version_dtype = dtype, out_dtype
        meta = get_meta(
            "bsr_dense_addmm",
            key,
            device_name,
            version=(_version, version_dtype, sparsity),
        )
        if meta is None and sparsity != 0.5:
            meta = get_meta(
                "bsr_dense_addmm",
                key,
                device_name,
                version=(_version, version_dtype, 0.5),
            )
        if meta is None and dtype is not out_dtype:
            meta = get_meta(
                "bsr_dense_addmm", key, device_name, version=(_version, dtype, 0.5)
            )
        if meta is None:
            # If still no meta found, search for approximate considering N divisibility
            approx_key = (*key[:2], "*", *key[3:-1], True)
            matching_meta = get_meta(
                "bsr_dense_addmm",
                approx_key,
                device_name,
                version=(_version, version_dtype, 0.5),
            )
            if matching_meta is None and dtype is not out_dtype:
                matching_meta = get_meta(
                    "bsr_dense_addmm",
                    approx_key,
                    device_name,
                    version=(_version, dtype, 0.5),
                )
            for mkey in sorted(matching_meta or {}):
                meta_ = matching_meta[mkey]
                n = mkey[2]
                split_n = meta_["SPLIT_N"]
                c = n // split_n
                if N % c == 0 and n <= N:
                    meta = dict(meta_)
                    meta["SPLIT_N"] = N // c
                    break
        if meta is not None:
            meta.update(**extra)
            return meta
        else:
            warn_once(
                "bsr_dense_addmm uses non-optimal triton kernel parameters"
                f" for {M=} {K=} {N=} {BM=}, {BK=} {beta=} {alpha=} {dtype=} {out_dtype=}. "
                "To find optimal triton kernel parameters, run with BSR_AUTOTUNE=1"
            )

    SPLIT_N = SPLIT_N or max(N // BM, 1)
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    num_stages = num_stages or 4
    num_warps = num_warps or 4
    return dict(
        SPLIT_N=SPLIT_N,
        GROUP_SIZE_ROW=GROUP_SIZE_ROW,
        num_stages=num_stages,
        num_warps=num_warps,
        **extra,
    )


def bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=1,
    alpha=1,
    left_alpha: Optional[torch.Tensor] = None,
    right_alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[tuple[Optional[int], Optional[int], Optional[int]]] = None,
    meta: Optional[dict] = None,
):
    """Compute

      out = beta * input + left_alpha.reshape(-1, 1) * (alpha * (bsr @ dense)) * right_alpha.reshape(1, -1)

    where left_alpha, right_alpha are (* + 1)-D tensors when
    specified, otherwise, these are treated as tensors filled with
    ones.
    """
    f_name = "bsr_dense_addmm"
    values = bsr.values()
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    batch_ndim = crow_indices.dim() - 1
    M, K = bsr.shape[batch_ndim : batch_ndim + 2]
    BM, BK = values.shape[batch_ndim + 1 : batch_ndim + 3]
    N = dense.shape[-1]

    original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
    if out is None:
        out = dense.new_empty(original_batch_dims_broadcasted + (M, N))

    if bsr._nnz() == 0 or alpha == 0 or N == 0 or M == 0 or K == 0:
        if beta == 0:
            out.zero_()
        else:
            out.copy_(input)
            if beta != 1:
                out.mul_(beta)
        return out

    if meta is None:
        sparsity = round(1 - bsr._nnz() * BM * BK / (M * K), 2)
        if AUTOTUNE:
            meta = tune_bsr_dense_addmm(
                input,
                bsr,
                dense,
                beta=beta,
                alpha=alpha,
                left_alpha=left_alpha,
                right_alpha=right_alpha,
                out=out,
                store=True,
                force=False,
                verbose=True,
                opname="bsr_dense_addmm",
            )
        else:
            meta = bsr_dense_addmm_meta(
                M,
                K,
                N,
                (BM, BK),
                beta,
                alpha,
                sparsity=sparsity,
                dtype=dense.dtype,
                out_dtype=out.dtype,
            )

    left_alpha_is_one = False
    right_alpha_is_one = False
    if left_alpha is None:
        left_alpha_is_one = True
        left_alpha = dense.new_empty(()).expand(
            *original_batch_dims_broadcasted, M, N
        )  # not referenced
    else:
        left_alpha = left_alpha.view(*original_batch_dims_broadcasted, M, 1).expand(
            *original_batch_dims_broadcasted, M, N
        )

    if right_alpha is None:
        right_alpha_is_one = True
        right_alpha = dense.new_empty(()).expand(
            *original_batch_dims_broadcasted, M, N
        )  # not referenced
    else:
        right_alpha = right_alpha.view(*original_batch_dims_broadcasted, 1, N).expand(
            *original_batch_dims_broadcasted, M, N
        )
    assert left_alpha.stride()[-1] == 0
    assert right_alpha.stride()[-2] == 0

    out_backup = out

    (
        crow_indices,
        col_indices,
        values,
        input,
        dense,
        left_alpha,
        right_alpha,
        out,
    ) = prepare_inputs(bsr, input, dense, left_alpha, right_alpha, out)

    SPLIT_N = meta.get("SPLIT_N", max(N // BM, 1))
    BN = N // SPLIT_N

    if N % SPLIT_N != 0:
        raise ValueError(
            f"bsr_dense_addmm only supports N divisible by {SPLIT_N}, got {N}, {SPLIT_N}"
        )

    out_untiled = out
    out = tile_to_blocksize(out, (BM, BN))
    dense = tile_to_blocksize(dense, (BK, BN))
    input = tile_to_blocksize(input, (BM, BN))
    left_alpha = tile_to_blocksize(left_alpha, (BM, BN))
    right_alpha = tile_to_blocksize(right_alpha, (BM, BN))

    # Determine accumulator type based on output dtype
    dot_out_dtype = {
        torch.float16: tl.float32,
        torch.bfloat16: tl.float32,
        torch.float32: tl.float64,
        torch.float64: tl.float64,
        torch.int8: tl.int32,
        torch.int32: tl.int32,
    }[out.dtype]

    n_batches = dense.size(0)
    n_block_rows = crow_indices.size(-1) - 1
    n_block_cols = dense.size(-3)

    full_grid = (n_batches, n_block_cols, n_block_rows)
    if max_grid is not None:
        grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
    else:
        grid_blocks = None

    tensor_dims_map = {
        values: (0, None, None),        # 1, 2144, 64, 8
        crow_indices: (0, None, -1),    # 1, 33
        col_indices: (0, None, None),   # 1, 2144
        input: (0, -3, -4),             # 1, 32, 16, 64, 64
        dense: (0, -3, None),           # 1, 128, 16, 8, 64
        left_alpha: (0, -3, -4),        # 1, 32, 16, 64, 64
        right_alpha: (0, -3, -4),       # 1, 32, 16, 64, 64
        out: (0, -3, -4),               # 1, 32, 16, 64, 64
    }

    assert alpha != 0

        
    def kernel(grid, *sliced_tensors):
        _bsr_strided_addmm_kernel[grid](
            *ptr_stride_extractor(*sliced_tensors),
            beta,
            alpha,
            beta_is_one=beta == 1,
            beta_is_nonzero=beta != 0,
            alpha_is_one=alpha == 1,
            left_alpha_is_one=left_alpha_is_one,
            right_alpha_is_one=right_alpha_is_one,
            BLOCKSIZE_ROW=BM,
            BLOCKSIZE_INNER=BK,
            BLOCKSIZE_COL=BN,
            allow_tf32=dot_out_dtype == tl.float32,
            acc_dtype=dot_out_dtype,
            **meta,
        )

    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    if out.data_ptr() != out_backup.data_ptr():
        # prepare_inputs has made a copy of out, copy its content back
        # to out_backup:
        out_backup.copy_(out_untiled.view(out_backup.shape))

    return out_backup


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _bsr_strided_addmm_kernel(
        # values prologue
        values_ptr,
        values_batch_stride,
        values_nnz_stride,
        values_row_block_stride,
        values_col_block_stride,
        # values epilogue
        # crow_indices prologue
        crow_indices_ptr,
        crow_indices_batch_stride,
        crow_indices_stride,
        # crow_indices epilogue
        # col_indices prologue
        col_indices_ptr,
        col_indices_batch_stride,
        col_indices_stride,
        # col_indices epilogue
        # input prologue
        input_ptr,
        input_batch_stride,
        input_tiled_row_stride,
        input_tiled_col_stride,
        input_row_block_stride,
        input_col_block_stride,
        # input epilogue
        # dense prologue
        dense_ptr,
        dense_batch_stride,
        dense_tiled_row_stride,
        dense_tiled_col_stride,
        dense_row_block_stride,
        dense_col_block_stride,
        # dense epilogue
        # left_alpha prologue
        left_alpha_ptr,
        left_alpha_batch_stride,
        left_alpha_tiled_row_stride,
        left_alpha_tiled_col_stride: tl.constexpr,
        left_alpha_row_block_stride,
        left_alpha_col_block_stride: tl.constexpr,
        # left_alpha epilogue
        # right_alpha prologue
        right_alpha_ptr,
        right_alpha_batch_stride,
        right_alpha_tiled_row_stride: tl.constexpr,
        right_alpha_tiled_col_stride,
        right_alpha_row_block_stride: tl.constexpr,
        right_alpha_col_block_stride,
        # right_alpha epilogue
        # output prologue
        output_ptr,
        output_batch_stride,
        output_tiled_row_stride,
        output_tiled_col_stride,
        output_row_block_stride,
        output_col_block_stride,
        # output epilogue
        beta,
        alpha,
        beta_is_one: tl.constexpr,
        beta_is_nonzero: tl.constexpr,
        alpha_is_one: tl.constexpr,
        left_alpha_is_one: tl.constexpr,
        right_alpha_is_one: tl.constexpr,
        BLOCKSIZE_ROW: tl.constexpr,
        BLOCKSIZE_COL: tl.constexpr,
        BLOCKSIZE_INNER: tl.constexpr,
        acc_dtype: tl.constexpr,
        allow_tf32: tl.constexpr,
        GROUP_SIZE_ROW: tl.constexpr,
        SPLIT_N: tl.constexpr,
    ):
        MIN_BLOCK_SIZE: tl.constexpr = 16   
        
        # left/right_alpha tensors are originally (* + 1)-dimensional
        assert left_alpha_tiled_col_stride == 0
        assert left_alpha_col_block_stride == 0
        assert right_alpha_tiled_row_stride == 0
        assert right_alpha_row_block_stride == 0

        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)

        row_block_pid, col_block_pid = tl.swizzle2d(
            row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW
        )

        crow_indices_offset_ptr = (
            crow_indices_ptr
            + crow_indices_batch_stride * batch_pid
            + crow_indices_stride * row_block_pid
        )
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

        # Compute nnz for the row with number row_block_pid.
        row_nnz = nnz_offset_next - nnz_offset

        #---Set up padding for block sizes<MIN BLOCK SIZE -
        if BLOCKSIZE_ROW < MIN_BLOCK_SIZE:
            PADDED_BLOCKSIZE_ROW:tl.constexpr = MIN_BLOCK_SIZE
        else:
            PADDED_BLOCKSIZE_ROW:tl.constexpr = BLOCKSIZE_ROW

        if BLOCKSIZE_INNER < MIN_BLOCK_SIZE:
            PADDED_BLOCKSIZE_INNER:tl.constexpr = MIN_BLOCK_SIZE
        else:
            PADDED_BLOCKSIZE_INNER:tl.constexpr = BLOCKSIZE_INNER

        if BLOCKSIZE_COL < MIN_BLOCK_SIZE or BLOCKSIZE_COL % MIN_BLOCK_SIZE != 0:
            PADDED_BLOCKSIZE_COL: tl.constexpr = (BLOCKSIZE_COL + MIN_BLOCK_SIZE - 1) // MIN_BLOCK_SIZE * MIN_BLOCK_SIZE
        else:
            PADDED_BLOCKSIZE_COL: tl.constexpr = BLOCKSIZE_COL


        row_block_arange = tl.arange(0, PADDED_BLOCKSIZE_ROW)
        inner_block_arange = tl.arange(0, PADDED_BLOCKSIZE_INNER)
        col_block_arange = tl.arange(0, PADDED_BLOCKSIZE_COL)

        # Initialize pointers
        values_block_ptrs = (
            values_ptr
            + values_batch_stride * batch_pid
            + values_nnz_stride * nnz_offset
            + values_row_block_stride * row_block_arange[:, None]
            + values_col_block_stride * inner_block_arange[None, :]
        )

        #Mask for loading values(handle row and inner padding)
        values_load_mask = (row_block_arange[:,None] < BLOCKSIZE_ROW) & \
            (inner_block_arange[None,:] < BLOCKSIZE_INNER)
        
        dense_block_ptrs = (
            dense_ptr
            + dense_batch_stride * batch_pid
            + dense_tiled_col_stride * col_block_pid
            + dense_row_block_stride * inner_block_arange[:, None]
            + dense_col_block_stride * col_block_arange[None, :]
        )

        # Mask for loading dense (handle inner and col padding)
        dense_load_mask = (inner_block_arange[:,None] < BLOCKSIZE_INNER) & \
            (col_block_arange[None,:] < BLOCKSIZE_COL)
        
        # Output pointers set to exact write locations for the current block
        output_ptrs = (
            output_ptr
            + output_batch_stride * batch_pid
            + output_tiled_row_stride * row_block_pid
            + output_tiled_col_stride * col_block_pid
            + output_row_block_stride * row_block_arange[:, None]
            + output_col_block_stride * col_block_arange[None, :]
        )

        # Mask for storing output(handle row and col padding)
        output_store_mask = (row_block_arange[:,None] < BLOCKSIZE_ROW) & \
            (col_block_arange[None,:] < BLOCKSIZE_COL)
        
        # Set pointer to the first nonzero element in the current row
        col_index_nnz_ptr = (
            col_indices_ptr
            + col_indices_batch_stride * batch_pid
            + col_indices_stride * nnz_offset
        )

        output_acc_block = tl.zeros(
            (PADDED_BLOCKSIZE_ROW, PADDED_BLOCKSIZE_COL), dtype=acc_dtype
        )
        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs, mask=values_load_mask, other=0.0)

            dense_row_idx = tl.load(col_index_nnz_ptr)
            # Load dense block with inner and col padding mask
            dense_block = tl.load(
                dense_block_ptrs + dense_tiled_row_stride * dense_row_idx,
                mask = dense_load_mask,
                other=0.0
            )

            # do block mm: tl.dot inputs now have logical shapes
            # (PADDED BLOCKSIZE ROW，PADDED BLOCKSIZE INNER) and
            # (PADDED BLOCKSIZE INNER,PADDED BLOCKSIZE cOL), satisfying the assertion.
            output_acc_block += tl.dot(
                values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype
            )

            # move val/col_index ptrs to the next block in the row
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

        if not alpha_is_one:
            output_acc_block *= alpha

        if not left_alpha_is_one:
            left_alpha_load_mask = row_block_arange[:,None] < BLOCKSIZE_ROW
            left_alpha_ptrs = (
                left_alpha_ptr
                + left_alpha_batch_stride * batch_pid
                + left_alpha_tiled_row_stride * row_block_pid
                + left_alpha_tiled_col_stride * col_block_pid
                + left_alpha_row_block_stride * row_block_arange[:, None]
                + left_alpha_col_block_stride * col_block_arange[None, :]
            )
            output_acc_block *= tl.load(left_alpha_ptrs, mask=left_alpha_load_mask, other=1.0)

        if not right_alpha_is_one:
            right_alpha_load_mask = col_block_arange[None,:] < BLOCKSIZE_COL
            right_alpha_ptrs = (
                right_alpha_ptr
                + right_alpha_batch_stride * batch_pid
                + right_alpha_tiled_row_stride * row_block_pid
                + right_alpha_tiled_col_stride * col_block_pid
                + right_alpha_row_block_stride * row_block_arange[:, None]
                + right_alpha_col_block_stride * col_block_arange[None, :]
            )
            output_acc_block *= tl.load(right_alpha_ptrs, mask=right_alpha_load_mask, other=1.0)

        if beta_is_nonzero:
            input_load_mask = (
                row_block_arange[:,None] < BLOCKSIZE_ROW & \
                col_block_arange[None,:] < BLOCKSIZE_COL 
            )
            input_ptrs = (
                input_ptr
                + input_batch_stride * batch_pid
                + input_tiled_row_stride * row_block_pid
                + input_tiled_col_stride * col_block_pid
                + input_row_block_stride * row_block_arange[:, None]
                + input_col_block_stride * col_block_arange[None, :]
            )
            input_block = tl.load(input_ptrs, mask=input_load_mask, other=0.0)
            if beta_is_one:
                output_acc_block += input_block
            else:
                output_acc_block += beta *input_block

        # write back the result
        tl.store(
            output_ptrs,
            output_acc_block.to(output_ptr.dtype.element_ty),
            mask=output_store_mask,
        )

else:
    _bsr_strided_addmm_kernel = None  # type: ignore[assignment]
