import triton
import triton.language as tl
from triton import Config

# TODO: add early config prune and estimate_matmul_time to reduce autotuning time
# from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
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
                    for split_k in [2, 4, 8, 16]:
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


def get_configs_compute_bound():
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


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


MIXED_MM_HEURISTICS = {
    "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    "BLOCK_K": lambda args: min(args["BLOCK_K"], args["QGROUP_SIZE"])
    if not args["TRANSPOSED"]
    else args["BLOCK_K"],
    "BLOCK_N": lambda args: min(args["BLOCK_N"], args["QGROUP_SIZE"])
    if args["TRANSPOSED"]
    else args["BLOCK_N"],
    "SPLIT_K": lambda args: 1
    if args["IS_BFLOAT16"]
    else args["SPLIT_K"],  # atomic add not supported for bfloat16
}


@triton.jit
def _mixed_mm_kernel(
    # Operands
    A,
    B,
    scales_ptr,
    zeros_ptr,
    C,
    # Matrix dims.
    M,
    N,
    K,
    # a, b, c, scales / zeros strides
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    stride_scale_k,
    stride_scale_n,
    # Meta-params
    IS_BFLOAT16: tl.constexpr,
    QGROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    TRANSPOSED: tl.constexpr = False,
    GROUP_M: tl.constexpr = 8,
    # tl.dot options
    acc_dtype: tl.constexpr = tl.float32,
    input_precision: tl.constexpr = "ieee",
    fp8_fast_accum: tl.constexpr = False,
    # Only used for debugging
    DEBUG: tl.constexpr = False,
):
    """Mixed matmul kernel

    A has shape (M, K) and is float16, bfloat16, or float32

    B is i4 / s4 and has shape (K // 2, N) and is packed as uint8 / int8. See `packed_2xint4` for details.

    Scales and zeros are of shape (NUM_GROUPS, N) and are same dtype as A, where NUM_GROUPS = (K // QGROUP_SIZE)
    QGROUP_SIZE should be a multiple of BLOCK_K such that a vector of scales / zeros is loaded and broadcasted to block shape
    per mainloop iteration.

    In the transposed case, A is M x N and B is K x N, and we reduce along "N":
    - TLDR: we are loading rows of A and B blocks at a time, dequantizing and transposing each block of B to achieve the overall
    effect of a transposed matmul. This is necessary to perform a transposed matmul without unpacking and repacking the B matrix.
        - Indexing remains the same for A (the reduction dim (BLK_K / K) corresponds to axis 1 of A -- "N" above)
            - We load a BLK_M x BLK_K block of A
        - Indexing for B is now flipped: N <-> K
            - We load BLK_N x BLK_K block of B (remembering that the reduction dimension is axis 1 of B)
            - We dequantize and transpose to BLK_K x BLK_N
            - scale / zero indexing also change, since we are now iterating along the non-grouping dim within the mac loop and along
            the grouping dim across blocks.
        - Each mac loop calculates BLK_M x BLK_N -> M x "N"(= K)
        - Within the mac loop for each block, we iterate along axis=1 for **both** A and B since axis = 1 is now the reduction dim for B.

    NOTE: Assumes that the quantization grouping was done along the K dimension originally (i.e., QGROUP_SIZE consecutive elements
    of original weight matrix in the K dimension were grouped together when calculating min / max scaling factors).
    """

    if not TRANSPOSED:
        tl.static_assert(QGROUP_SIZE % BLOCK_K == 0)
    else:
        tl.static_assert(QGROUP_SIZE % BLOCK_N == 0)

    # TODO: refactor swizzling to separate function
    # Threadblock swizzling
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    if not DEBUG:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm
    # rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    # rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rak = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    # BLOCK_K for b is effectively BLOCK_K // 2
    if not TRANSPOSED:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(
                tl.multiple_of(rn % N, BLOCK_N), BLOCK_N
            )  # rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            rbn = rn
        rbk = pid_z * BLOCK_K // 2 + tl.arange(0, BLOCK_K // 2)
    else:
        rn = (pid_n * BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N // 2), BLOCK_N // 2)
        else:
            rbn = rn
        rbk = rak

    A = A + (ram[:, None] * stride_am + rak[None, :] * stride_ak)

    if not TRANSPOSED:
        B = B + (rbk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    else:
        # Note: in the transposed case, we are loading BLK_N x BLK_K, but we need to transpose to BLK_K x BLK_N
        # the strides are adjusted accordingly, since we to stride by stride_bk to get rows of BLK_N
        # and stride_bn to get columns of BLK_K
        B = B + (rbn[:, None] * stride_bk + rbk[None, :] * stride_bn)

    # Grouping is along K, so in the forward pass, each block loads a row vector of BLK_K x BLK_N
    # where grouping varies along N, hence the mainloop marches down the K dimension, where
    # group idx is given by K // QGROUP_SIZE

    if not TRANSPOSED:
        offsets_scale_n = (
            pid_n * stride_scale_n * BLOCK_N + tl.arange(0, BLOCK_N) * stride_scale_n
        )
    else:
        scale_offset_k = pid_n * BLOCK_N * stride_scale_k // QGROUP_SIZE
        offsets_scale_n = tl.arange(0, BLOCK_K) * stride_scale_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            qb = tl.load(B)
        else:
            k_remaining_a = K - k * (BLOCK_K * SPLIT_K)
            if not TRANSPOSED:
                k_remaining_b = (
                    K - k * (BLOCK_K * SPLIT_K) // 2
                )  # Note the division by 2
            else:
                k_remaining_b = K - k * (BLOCK_K * SPLIT_K)  # = k_remaining_a

            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rak[None, :] < k_remaining_a, other=_0)
            qb = tl.load(B, mask=rbk[:, None] < k_remaining_b, other=_0)

        if not TRANSPOSED:
            scale_offset_k = k * BLOCK_K * SPLIT_K * stride_scale_k // QGROUP_SIZE
        else:
            offsets_scale_n = (
                k * stride_scale_n * BLOCK_K + tl.arange(0, BLOCK_K) * stride_scale_n
            )

        scales = tl.load(scales_ptr + offsets_scale_n + scale_offset_k)
        zeros = tl.load(zeros_ptr + offsets_scale_n + scale_offset_k)

        # Unpack qweights -- h/t jlebar!
        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        qb_lo = (qb << _4_i8) >> _4_i8
        qb_hi = qb >> _4_i8

        # Upcast to fp16
        # TODO: better bfloat16 conversion? compilation error if direct conversion from int8 to bfloat16
        if IS_BFLOAT16:
            dq_b = (
                tl.join(
                    qb_lo.to(tl.float16).to(A.dtype.element_ty),
                    qb_hi.to(tl.float16).to(A.dtype.element_ty),
                ).permute(0, 2, 1)
                # .reshape(BLOCK_K, BLOCK_N)
            )
        else:
            dq_b = (
                tl.join(
                    qb_lo.to(A.dtype.element_ty),
                    qb_hi.to(A.dtype.element_ty),
                ).permute(0, 2, 1)
                # .reshape(BLOCK_K, BLOCK_N)
            )
        if not TRANSPOSED:
            dq_b = dq_b.reshape(BLOCK_K, BLOCK_N)
        else:
            dq_b = dq_b.reshape(BLOCK_N, BLOCK_K)

        # Scale upcasted weights
        # Note that we broadcast the scales --> the assumption is that all scales fall within a single QGROUP
        # This condition is statically check (see assertions above)

        zeros = zeros[None, :]
        scales = scales[None, :]

        dq_b = (dq_b - zeros) * scales

        if TRANSPOSED:
            dq_b = tl.trans(dq_b)

        if fp8_fast_accum:
            acc = tl.dot(
                a, dq_b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, dq_b, out_dtype=acc_dtype, input_precision=input_precision)
        A += BLOCK_K * SPLIT_K * stride_ak

        # Advance by half the block size, since each block is unpacked and upcasted into two fp16 values
        if not TRANSPOSED:
            B += BLOCK_K * SPLIT_K * stride_bk // 2
        else:
            # we iterating across a row of B (non-packing dim, hence no need for div 2)
            B += BLOCK_K * SPLIT_K * stride_bn
    acc = acc.to(C.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_mixed_mm = triton.heuristics(MIXED_MM_HEURISTICS)(_mixed_mm_kernel)
mixed_mm_kernel_max_autotune = triton.autotune(
    configs=get_configs_compute_bound() + get_configs_io_bound(), key=["M", "N", "K"]
)(_mixed_mm)
mixed_mm_kernel_compute_bound = triton.autotune(
    configs=get_configs_compute_bound(), key=["M", "N", "K"]
)(_mixed_mm)
_mixed_mm_debug = _mixed_mm
