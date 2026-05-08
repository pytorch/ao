"""
Triton kernel for Randomized Hadamard Transform (RHT) with fused global amax reduction.

Entry point: triton_rht_amax(A) returns a scalar float32 global absolute maximum of
the post-RHT output without materializing the full (N, M) output tensor. Uses a
persistent warp-specialized TMA kernel with per-CTA cumulative max and one atomic_max
per CTA into a caller-provided scalar buffer.
"""

import itertools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from torchao.prototype.mx_formats.hadamard_utils import _compute_pid, get_rht_matrix
from torchao.utils import is_sm_at_least_100

# SM100+ autotune configs. BLOCK_M must be divisible by 16 (RHT reshape constraint).
HADAMARD_TILE_SHAPES: list[tuple[int, int]] = [
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 32),
    (128, 64),
]

HADAMARD_CONFIGS: list[triton.Config] = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_STAGES": ns},
        num_warps=nw,
        num_stages=ns,
    )
    for (bm, bn), ns, nw in itertools.product(
        HADAMARD_TILE_SHAPES,
        [2, 3, 4],  # NUM_STAGES
        [4, 8],  # NUM_WARPS
    )
]


@triton.autotune(configs=HADAMARD_CONFIGS, key=["M", "N"])
@triton.jit
def _hadamard_amax_kernel(
    a_ptr,
    b_ptr,
    global_rht_amax_ptr,
    global_a_amax_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Persistent RHT kernel with fused amax reduction; no output tensor written."""
    # Create TMA descriptors in-kernel from raw pointers and shape/stride
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[16, 16],
        strides=[16, 1],
        block_shape=[16, 16],
    )

    # Persistent grid-stride loop
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    num_tiles = num_pid_m * num_pid_n

    # Load (16, 16) random hadamard matrix once
    hadamard = b_desc.load([0, 0])

    # Track cumulative max across all tiles for this block
    cumulative_rht_amax = tl.zeros((BLOCK_N * BLOCK_M // 16, 16), dtype=tl.float32)
    cumulative_a_amax = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # warp-specialized: producer warps issue TMA loads, consumer warps run wgmma
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=True,
        num_stages=NUM_STAGES,
    ):
        pid_n, pid_m = _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N)

        # Load A (BLOCK_M, BLOCK_N)
        a = a_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])

        # Transpose A_t (BLOCK_N, BLOCK_M)
        a_t = tl.trans(a)

        # Reshape to A_r (BLOCK_N * BLOCK_M//16, 16)
        a_t_r = tl.reshape(a_t, [BLOCK_N * BLOCK_M // 16, 16])

        a_t_rht = tl.dot(a_t_r, hadamard)

        # Cast to bfloat16 like regular matmul output
        a_t_rht = a_t_rht.to(tl.bfloat16)

        # Update cumulative max at tile level to avoid failing
        # TritonGPUAutomaticWarpSpecialization MLIR pass
        abs_a_t_rht = tl.abs(a_t_rht)
        cumulative_rht_amax = tl.maximum(cumulative_rht_amax, abs_a_t_rht)

        cumulative_a_amax = tl.maximum(cumulative_a_amax, tl.abs(a.to(tl.float32)))

    # Get scalar max for this block and update global max with atomic max operation
    tile_rht_amax = tl.max(tl.max(cumulative_rht_amax, axis=1), axis=0)
    tl.atomic_max(global_rht_amax_ptr, tile_rht_amax.to(tl.float32))

    tile_a_amax = tl.max(tl.max(cumulative_a_amax, axis=1), axis=0)
    tl.atomic_max(global_a_amax_ptr, tile_a_amax.to(tl.float32))


def triton_rht_amax(
    A: torch.Tensor,
    sign_vector: tuple[int, ...] | None = None,
    hadamard_dimension: int = 16,
    scaling_type: F.ScalingType = F.ScalingType.TensorWise,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RHT to A and return global absolute maxima without materializing output.

    Args:
        A: (M, N) bfloat16 tensor, row-major. M must be divisible by 16.
        sign_vector: Optional sign vector for the RHT. If None, a random one is generated.
        hadamard_dimension: Dimension of the Hadamard matrix (default 16).
        scaling_type: ScalingType controlling reduction granularity. Only
            ``ScalingType.TensorWise`` is currently supported.

    Returns:
        Tuple of (global_rht_amax, global_a_amax):
          - global_rht_amax: scalar float32 containing max(abs(RHT(A))).
          - global_a_amax: scalar float32 containing max(abs(A)).

    Raises:
        NotImplementedError: If hardware is pre-SM100.
        ValueError: If A is not bfloat16, not 2-D, not contiguous, M % 16 != 0, or
            scaling_type is not ScalingType.TensorWise.
    """
    if torch.cuda.is_available() and not is_sm_at_least_100():
        raise NotImplementedError(
            "Kernel requires SM100 (Blackwell); detected pre-SM100 hardware."
        )
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 2:
        raise ValueError("Tensor A must be 2-D")
    if not A.is_contiguous():
        raise ValueError("A must be row-major (contiguous)")
    if A.shape[0] % 16 != 0:
        raise ValueError(f"M must be divisible by 16, got M={A.shape[0]}")
    if scaling_type != F.ScalingType.TensorWise:
        raise ValueError(
            f"scaling_type={scaling_type!r} is not supported; "
            "only ScalingType.TensorWise is implemented."
        )
    M, N = A.shape

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    GROUP_SIZE_N: int = 8  # L2 reuse grouping along M

    # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch space.
    # Outside torch.compile, none is set by default; mirror what torch._inductor does.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=A.device
            )
        )

    B = get_rht_matrix(
        sign_vector=sign_vector, device=A.device, hadamard_dimension=hadamard_dimension
    ).to(torch.bfloat16)
    global_rht_amax = torch.zeros((), dtype=torch.float32, device=A.device)
    global_a_amax = torch.zeros((), dtype=torch.float32, device=A.device)

    _hadamard_amax_kernel[(NUM_SMS,)](
        A,
        B,
        global_rht_amax,
        global_a_amax,
        M,
        N,
        GROUP_SIZE_N=GROUP_SIZE_N,
        NUM_SMS=NUM_SMS,
    )
    return global_rht_amax, global_a_amax
