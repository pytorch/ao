# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
import triton
import triton.language as tl

from torchao.prototype.common.triton.matmul_perf_model import (
    early_config_prune,
    estimate_matmul_time,
)

from .adam_step import BETA1, BETA2, EPS
from .custom_autotune import Config, autotune
from .matmul import (
    TRITON_ACC_TYPES,
    get_higher_dtype,
    get_mm_heuristics,
    init_to_zero,
    to_tl_type,
)
from .matmul import get_autotuner as default_mm_autotuner

logger = logging.getLogger(__name__)

AUTOTUNER_TOP_K = 50


def set_tuner_top_k(k):
    global AUTOTUNER_TOP_K
    AUTOTUNER_TOP_K = k


@triton.jit
def _fused_adam_mm_kernel(
    # matmul args
    A,
    B,
    C,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    # adam epilogue,
    exp_avg_ptr,  # these will be updated inplace
    exp_avg2_ptr,
    store,
    # grad_ptr,  # low rank grad output -- not needed, C is the output
    # meta params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    # Adam-specific params
    BETA1: tl.constexpr = BETA1,
    BETA2: tl.constexpr = BETA2,
    EPS: tl.constexpr = EPS,
    # matmul kernel settings
    acc_dtype: tl.constexpr = tl.float32,  #
    allow_tf32: tl.constexpr = False,  # higher precision for this phase
    fp8_fast_accum: tl.constexpr = False,  #
    AB_DTYPE: tl.constexpr = None,  #
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    # acc = acc.to(C.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    epilogue_offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    # Load adam state
    exp_avg = tl.load(exp_avg_ptr + epilogue_offsets, mask=mask)
    exp_avg2 = tl.load(exp_avg2_ptr + epilogue_offsets, mask=mask)

    # Perform update
    exp_avg = BETA1 * exp_avg.to(acc.dtype) + (1.0 - BETA1) * acc
    exp_avg2 = BETA2 * exp_avg2.to(acc.dtype) + (1.0 - BETA2) * (acc * acc)
    denom = tl.sqrt(exp_avg2) + EPS
    norm_grad = exp_avg / denom
    # Convert to output type
    norm_grad = norm_grad.to(C.dtype.element_ty)

    # acc = acc.to(C.dtype.element_ty)
    C = C + epilogue_offsets

    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, norm_grad, mask=mask)
    else:
        tl.atomic_add(C, norm_grad, mask=mask)

    if store:
        tl.store(
            exp_avg_ptr + epilogue_offsets,
            exp_avg,
            mask=mask,
        )
        tl.store(
            exp_avg2_ptr + epilogue_offsets,
            exp_avg2,
            mask=mask,
        )


def _get_configs_splitk_all():
    """
    Configs specific to split-k matmuls
    Not used currently
    """
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128]:
            for block_k in [16, 32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in [2, 4, 8]:
                        configs.append(
                            Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            )
                        )
    return configs


def _get_configs_splitk_small():
    """Configs for split-k, smaller version than above
    Not used currently
    """
    configs = []
    for num_stages in [2, 3, 4]:
        for block_m in [64, 128]:
            for block_k in [16, 32, 64]:
                for block_n in [64, 128]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in [2, 4, 8]:
                        configs.append(
                            Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            )
                        )
    return configs


def _splitk_autotuner(
    configs=_get_configs_splitk_small(),
    key=["M", "N", "K"],
    early_config_prune=early_config_prune,
    perf_model=estimate_matmul_time,
    top_k=AUTOTUNER_TOP_K,
):
    """Autotuner for splitk matmuls
    Not used currently
    """
    autotuner = autotune(
        configs=configs,
        key=key,
        prune_configs_by={
            "early_config_prune": early_config_prune,
            "perf_model": perf_model,
            "top_k": top_k,
        },
    )

    return autotuner


def _get_kernel(
    tuner_fn=default_mm_autotuner, heuristics_fn=get_mm_heuristics, topk=AUTOTUNER_TOP_K
):
    tuner = tuner_fn()
    tuner.topk = topk
    heuristics = heuristics_fn()
    return tuner(heuristics(_fused_adam_mm_kernel))


DEFAULT_KERNEL = _get_kernel()


def fused_adam_mm_launcher(
    a,
    b,
    *,
    exp_avg,
    exp_avg2,
    store=True,
    BETA1=BETA1,
    BETA2=BETA2,
    EPS=EPS,
    allow_tf32=False,
    fp8_fast_accum=False,
    acc_dtype=None,
    output_dtype=None,
    kernel=None,
):
    device = a.device
    # handle non-contiguous inputs if necessary
    # a = grad
    # b = galore_proj.ortho_matrix.t()
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # common type between a and b
    ab_dtype = get_higher_dtype(a.dtype, b.dtype)

    # allocates output
    if output_dtype is None:
        output_dtype = ab_dtype

    c = torch.empty((M, N), device=device, dtype=output_dtype)

    if acc_dtype is None:
        acc_dtype = [ab_dtype][0]
    else:
        assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
        assert (
            acc_dtype in TRITON_ACC_TYPES[a.dtype]
        ), "acc_dtype not compatible with the type of a"
        assert (
            acc_dtype in TRITON_ACC_TYPES[b.dtype]
        ), "acc_dtype not compatible with the type of b"

    acc_dtype = to_tl_type(acc_dtype)
    ab_dtype = to_tl_type(ab_dtype)
    output_dtype = to_tl_type(output_dtype)

    # Tensor cores support input with mixed float8 types.
    if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [
        tl.float8e4nv,
        tl.float8e5,
    ]:
        ab_dtype = None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    if kernel is None:
        kernel = DEFAULT_KERNEL
    kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        exp_avg,
        exp_avg2,
        store=store,
        BETA1=BETA1,  # ,  #
        BETA2=BETA2,  # ,  #
        EPS=EPS,  #
        acc_dtype=acc_dtype,  #
        allow_tf32=allow_tf32,  #
        fp8_fast_accum=fp8_fast_accum,  #
        GROUP_M=8,
        AB_DTYPE=ab_dtype,
    )
    return exp_avg, exp_avg2, c  # c -> normalized low rank grad
