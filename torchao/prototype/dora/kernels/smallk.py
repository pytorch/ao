import heapq
import logging
from enum import Enum, StrEnum, unique

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from torchao.prototype.common.triton.matmul import (
    estimate_matmul_time,
    get_configs_io_bound,
    get_higher_dtype,
)

from .custom_autotune import Config, autotune

logger = logging.getLogger(__name__)


@unique
class SwizzleType(Enum):
    GROUPED = 0
    COLUMN_MAJOR = 1
    ROW_MAJOR = 2


class TritonInputPrecision(StrEnum):
    IEEE: str = "ieee"
    TF32: str = "tf32"
    TF32X3: str = "tf32x3"


TRITON_SUPPORTED_ACC_TYPES = {
    torch.float16: (torch.float32, torch.float16),
    torch.bfloat16: (torch.float32, torch.bfloat16),
    torch.float32: (torch.float32,),
    torch.int8: (torch.int32,),
}


def to_tl_type(ty):
    return getattr(tl, str(ty).split(".")[-1])


def get_compute_bound_configs():
    configs = [
        # basic configs for compute-bound matmuls
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
        # good for int8
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
    ]
    return configs


@triton.jit()
def swizzle_tile(
    pid,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    SWIZZLE: tl.constexpr,
):
    if SWIZZLE == tl.constexpr(SwizzleType.GROUPED):
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
    else:
        tl.static_assert(False, "swizzle type not supported")

    return pid_m, pid_n


def get_small_k_configs():
    configs = get_compute_bound_configs() + get_configs_io_bound()
    KEYS_TO_REMOVE = ["BLOCK_K", "SPLIT_K"]
    for cfg in configs:
        for key in KEYS_TO_REMOVE:
            del cfg.kwargs[key]

    return configs


def small_k_early_config_prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    dtsize = named_args["A"].element_size()

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            named_args["K"],
            config.num_stages,
        )

        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs

    # Some dtypes do not allow atomic_add
    # if dtype not in [torch.float16, torch.float32]:
    #     configs = [config for config in configs if config.kwargs["SPLIT_K"] == 1]

    # group configs by (BLOCK_M,_N,_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            named_args["K"],
            # kw["SPLIT_K"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = k
        if capability[0] >= 8:
            # compute cycles (only works for ampere GPUs)
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: 10 + abs(x[1] - optimal_num_stages)
                if (x[1] - optimal_num_stages) < 0
                else x[1] - optimal_num_stages,
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs


SMALLK_HEURISTICS = {
    "BLOCK_K": lambda args: args["K"],
}

_AUTOTUNE_TOPK = 10


# @heuristics(SMALLK_HEURISTICS)
@autotune(
    get_small_k_configs(),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": small_k_early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": _AUTOTUNE_TOPK,
    },
)
@triton.jit
def _mm_small_k_kernel(
    A,
    B,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    acc_dtype: tl.constexpr,  #
    input_precision: tl.constexpr,  #
    fp8_fast_accum: tl.constexpr,  #
    BLOCK_K: tl.constexpr,  #
    AB_DTYPE: tl.constexpr,  #
    BLOCK_M: tl.constexpr = 256,
    BLOCK_N: tl.constexpr = 64,
    C=None,
    stride_cm=None,
    stride_cn=None,  #
    Norm2=None,
    Source=None,
    stride_sourcem=None,
    stride_sourcen=None,
    Magnitude=None,
    ADD_SOURCE: tl.constexpr = False,
    EPILOGUE_NORM: tl.constexpr = False,
    EPILOGUE_MAGNITUDE: tl.constexpr = False,
    STORE_ACC: tl.constexpr = False,
):
    pid_m = tl.program_id(0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    a = tl.load(A)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    rn = tl.arange(0, BLOCK_N)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    if STORE_ACC:
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)

    if ADD_SOURCE:
        Source = Source + (rm[:, None] * stride_sourcem + rn[None, :] * stride_sourcen)

    if EPILOGUE_NORM:
        norm_vec = tl.zeros((BLOCK_M,), dtype=acc_dtype)

    if EPILOGUE_MAGNITUDE:
        Magnitude = Magnitude + ram

    mask_m = rm < M

    for n in range(0, tl.cdiv(N, BLOCK_N)):
        # Advance B over N

        b = tl.load(B)

        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc = tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

        if ADD_SOURCE:
            mask_n = (n * BLOCK_N + rn < N)[None, :]
            source = tl.load(Source, mask=mask_m[:, None] & mask_n)
            acc += source.to(acc_dtype)
            Source += BLOCK_N * stride_sourcen

        # 2-norm = tl.sqrt(tl.sum(acc * acc, axis=1))
        if EPILOGUE_NORM:
            norm_vec += tl.sum(acc * acc, axis=1)

        if STORE_ACC:
            mask_n = (n * BLOCK_N + rn < N)[None, :]
            tl.store(C, acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n)
            C += BLOCK_N * stride_cn

        B += BLOCK_N * stride_bn

    if EPILOGUE_NORM:
        Norm2 = Norm2 + rm
        norm_vec = tl.rsqrt(norm_vec).to(Norm2.dtype.element_ty)

        if EPILOGUE_MAGNITUDE:
            magnitude = tl.load(Magnitude, mask=mask_m)
            norm_vec *= magnitude

        tl.store(Norm2, norm_vec, mask=mask_m)


def triton_mm_small_k(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_norm: bool = True,
    source: torch.Tensor = None,
    magnitude: torch.Tensor = None,
    store_acc: bool = False,
    acc_dtype: torch.dtype = None,
    input_precision: TritonInputPrecision = TritonInputPrecision.IEEE,
    fp8_fast_accum: bool = False,
    output_dtype: torch.dtype = None,
):
    """Computes GEMM for small K {16, 32, 64}

    Assumes that K is small enough that the MAC loop within each block is a single iteration.
    Instead of iterating over K, we iterate over N per block such that each block computes a BLK_M x N row of C.  Kernel grid is ceildiv(M, BLOCK_M).

    This specialized GEMM is primarily useful for low-rank projections and fusing grid-wide reductions into the epilogue.

    Currently, the following fusions are implemented:
    - `epilogue_norm` - when set to True, the kernel computes the reverse 2-norm along axis=1 of AB ( `1 / 2-norm(AB, axis=1)` )
    - `source=torch.Tensor` - when passed a tensor of shape `M x N`, the kernel computes `D = AB + source`
    - `magnitude=torch.Tensor` - when passed a tensor of shape `M`, the kernel additionally multiplies the epilogue norm by the magnitude vector

    Hence, when the above fusions are enabled, the kernel can be used to compute DoRA layer magnitude normalization: `magnitude * (base_weight + lora_B(lora_A(x))).norm(2, axis=1)`

    Args:
        a (torch.Tensor): operand A
        b (torch.Tensor): operand B
        source (torch.Tensor): Operand C in `D = AB + C`
        epilogue_norm (bool, optional): Whether to calculate 1 / 2-norm(AB, axis=1)
        magnitude (torch.Tensor): vector to multiply epilogue norm by
        store_acc (bool): whether to store `AB`, if False, then `epilogue_norm` must be True, in which case only the `2-norm` is stored
        acc_dtype (torch.DType): accumulator type in MAC loop
        input_precision (TritonInputPrecision): precision to use for fp32 matmul
        fp8_fast_accum (bool)
        output_dtype (torch.DType): type for output tensors (`D`, `2-norm`, etc.)

    Returns:
        torch.Tensor
    """
    assert store_acc or epilogue_norm, "Must use store_acc or epilogue_norm"

    device = a.device

    # Make sure inputs are contiguous
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "Incompatible operand dimensions"
    M, K = a.shape
    _, N = b.shape

    assert K < 128, "K must be < 128 to use this kernel"

    # common type between a and b
    ab_dtype = get_higher_dtype(a.dtype, b.dtype)

    if output_dtype is None:
        output_dtype = ab_dtype

    if epilogue_norm:
        norm2 = torch.zeros(M, device=device, dtype=output_dtype)

    # Must set out_dtype before converting dtypes to tl types
    if store_acc:
        c = torch.empty((M, N), device=device, dtype=output_dtype)

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

    # Convert dtypes to tl types
    acc_dtype = to_tl_type(acc_dtype)
    ab_dtype = to_tl_type(ab_dtype)
    output_dtype = to_tl_type(output_dtype)

    # Use fp8 types in MAC loop
    if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [
        tl.float8e4nv,
        tl.float8e5,
    ]:
        ab_dtype = None

    logger.debug(
        f"triton_mm_small_k: {ab_dtype=} {acc_dtype=} {input_precision=} {fp8_fast_accum=} {output_dtype=}"
    )

    # Set the fusion and other GEMM kwargs
    # IMPORTANT: BLOCK_K must be equal to K
    kwargs = {
        "BLOCK_K": K,
        "acc_dtype": acc_dtype,
        "input_precision": input_precision,
        "fp8_fast_accum": fp8_fast_accum,
        "AB_DTYPE": ab_dtype,
        "EPILOGUE_NORM": epilogue_norm,
        "ADD_SOURCE": source is not None,
        "EPILOGUE_MAGNITUDE": magnitude is not None,
        "STORE_ACC": store_acc,
    }

    # 2-norm params
    if epilogue_norm:
        kwargs["Norm2"] = norm2

    # source params
    if source is not None:
        assert source.shape == (M, N)
        kwargs["Source"] = source
        kwargs["stride_sourcem"] = source.stride(0)
        kwargs["stride_sourcen"] = source.stride(1)
    else:
        kwargs["Source"] = None
        kwargs["stride_sourcem"] = 0
        kwargs["stride_sourcen"] = 0

    # magnitude params, epilogue_norm must be True
    if magnitude is not None:
        assert epilogue_norm, "magnitude requires epilogue_norm"
        assert magnitude.ndim == 1 and magnitude.shape[0] == M
        kwargs["Magnitude"] = magnitude

    # store_acc, whether to store the intermediate AB
    if store_acc:
        kwargs["C"] = c
        kwargs["stride_cm"] = c.stride(0)
        kwargs["stride_cn"] = c.stride(1)
    else:
        kwargs["C"] = None
        kwargs["stride_cm"] = 0
        kwargs["stride_cn"] = 0

    # kwargs_str = " ".join(
    #     f"{k}={v}" for k, v in kwargs.items() if not isinstance(v, torch.Tensor)
    # )
    # print(f"triton_mm_small_k: {kwargs_str}")

    # launch kernel
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    _mm_small_k_kernel[grid](
        a,
        b,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        **kwargs,
    )

    if store_acc:
        if epilogue_norm:
            return c, norm2
        else:
            return c
    return norm2
