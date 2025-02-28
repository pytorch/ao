import logging

import torch
import triton
import triton.language as tl

from torchao.prototype.common.triton.matmul import (
    early_config_prune,
    estimate_matmul_time,
    get_configs_io_bound,
    get_higher_dtype,
)

from .common import (
    MATMUL_HEURISTICS,
    TRITON_SUPPORTED_ACC_TYPES,
    SwizzleType,
    TritonInputPrecision,
    get_compute_bound_configs,
    swizzle_tile,
    to_tl_type,
)
from .custom_autotune import autotune

logger = logging.getLogger(__name__)


_AUTOTUNE_TOPK = 10


@autotune(
    get_compute_bound_configs() + get_configs_io_bound(),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": _AUTOTUNE_TOPK,
    },
)
@triton.heuristics(
    {
        "EVEN_K": MATMUL_HEURISTICS["EVEN_K"],
        "SPLIT_K": MATMUL_HEURISTICS["SPLIT_K"],
    }
)
@triton.jit
def _matmul_kernel(
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
    acc_dtype: tl.constexpr,  #
    input_precision: tl.constexpr,  #
    fp8_fast_accum: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  #
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,  #
    SWIZZLE: tl.constexpr,
    EPILOGUE_ELEMENTWISE_ADD: tl.constexpr = False,
    Epilogue_source=None,
    EPILOGUE_BROADCAST_SCALE: tl.constexpr = False,
    Epilogue_scale=None,
):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    # Threadblock swizzle
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M, SWIZZLE)

    # Operand offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    # Operand pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    # Allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    # MAC Loop
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
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # Convert acc to output dtype
    acc = acc.to(C.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    # mask = (rm < M)[:, None] & (rn < N)[None, :]
    mask_m = (rm < M)[:, None]
    mask_n = (rn < N)[None, :]
    if EPILOGUE_ELEMENTWISE_ADD:
        Epilogue_source = Epilogue_source + (
            rm[:, None] * stride_cm + rn[None, :] * stride_cn
        )
        source = tl.load(Epilogue_source, mask=mask_m & mask_n)
        acc += source
    if EPILOGUE_BROADCAST_SCALE:
        Epilogue_scale = Epilogue_scale + (rn[None, :])
        scale = tl.load(Epilogue_scale, mask=mask_n)
        acc *= scale

    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask_m & mask_n)
    else:
        tl.atomic_add(C, acc, mask=mask_m & mask_n)


def triton_mm(
    a,
    b,
    epilogue_source=None,
    epilogue_scale=None,
    acc_dtype=None,
    input_precision=TritonInputPrecision.IEEE,
    fp8_fast_accum=False,
    output_dtype=None,
    swizzle: SwizzleType = SwizzleType.GROUPED,
    GROUP_M: int = 8,
):
    """Triton GEMM implementation, `D = AB + C`

    Based on `triton.ops.matmul`, with the addition of epilogue.

    Args:
        a (torch.Tensor): operand A
        b (torch.Tensor): operand B
        epilogue_source(optional, torch.Tensor): operand C in `D = AB + C`
        epilogue_scale(optional, torch.Tensor): row-wise scale-vector of dim `N` in `D = scale * (AB + C)`
        acc_dtype (torch.DType): accumulator type in MAC loop
        input_precision (TritonInputPrecision): precision to use for fp32 matmul
        fp8_fast_accum (bool)
        output_dtype (optional, torch.DType): output type of the GEMM, defaults to higher dtype of A / B

    Returns:
        torch.Tensor: `D = AB + C`
    """
    device = a.device
    # handle non-contiguous inputs if necessary
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

    # Epilogue pre-conditions
    # TODO Check strides?
    if epilogue_source is not None:
        assert epilogue_source.shape == (M, N), "incompatible dimensions"
        assert epilogue_source.dtype == c.dtype, "incompatible dtype"

    if epilogue_scale is not None:
        assert (
            epilogue_scale.ndim == 1 and epilogue_scale.shape[0] == N
        ), "incompatible dimensions"
        assert epilogue_scale.dtype == c.dtype, "incompatible dtype"

    # choose accumulator type
    if acc_dtype is None:
        acc_dtype = TRITON_SUPPORTED_ACC_TYPES[ab_dtype][0]
    else:
        assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
        assert (
            acc_dtype in TRITON_SUPPORTED_ACC_TYPES[a.dtype]
        ), "acc_dtype not compatible with the type of a"
        assert (
            acc_dtype in TRITON_SUPPORTED_ACC_TYPES[b.dtype]
        ), "acc_dtype not compatible with the type of b"

    # convert to triton types
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

    _matmul_kernel[grid](
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
        acc_dtype=acc_dtype,  #
        input_precision=input_precision,  #
        fp8_fast_accum=fp8_fast_accum,  #
        GROUP_M=GROUP_M,
        AB_DTYPE=ab_dtype,
        SWIZZLE=swizzle,
        EPILOGUE_ELEMENTWISE_ADD=epilogue_source is not None,
        Epilogue_source=epilogue_source,
        EPILOGUE_BROADCAST_SCALE=epilogue_scale is not None,
        Epilogue_scale=epilogue_scale,
    )
    return c
