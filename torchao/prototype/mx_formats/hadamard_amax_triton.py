"""
Triton kernel for Randomized Hadamard Transform (RHT) with fused global amax reduction.

Entry point: triton_rht_amax(A, sign_vector) returns a scalar float32 global
absolute maximum of the post-RHT output without materializing the full (N, M)
output tensor. Uses a persistent warp-specialized TMA kernel with per-CTA
cumulative max and one atomic_max per CTA into a caller-provided scalar buffer.
"""

import torch
import torch.nn.functional as F
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

_DEFAULT_SCALING_TYPE = (
    int(F.ScalingType.TensorWise) if hasattr(F, "ScalingType") else 0
)

if torch_version_at_least("2.10.0") and has_triton():
    import itertools
    from typing import List, Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.hadamard_utils import (
        _compute_pid,
        get_rht_matrix,
        prepare_for_cuda_graph,
    )
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
            pid_n, pid_m = _compute_pid(
                tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N
            )

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
            cumulative_rht_amax = tl.maximum(
                cumulative_rht_amax,
                abs_a_t_rht,
                propagate_nan=tl.PropagateNan.ALL,
            )

            abs_a = tl.abs(a.to(tl.float32))
            cumulative_a_amax = tl.maximum(
                cumulative_a_amax,
                abs_a,
                propagate_nan=tl.PropagateNan.ALL,
            )

        # Get scalar max for this block and update global max with atomic max operation
        tile_rht_amax = tl.max(tl.max(cumulative_rht_amax, axis=1), axis=0)
        tile_rht_has_nan = tl.max(
            tl.max((cumulative_rht_amax != cumulative_rht_amax).to(tl.int32), axis=1),
            axis=0,
        )
        tile_rht_amax = tl.where(tile_rht_has_nan != 0, float("nan"), tile_rht_amax)
        tl.atomic_max(global_rht_amax_ptr, tile_rht_amax.to(tl.float32))

        tile_a_amax = tl.max(tl.max(cumulative_a_amax, axis=1), axis=0)
        tile_a_has_nan = tl.max(
            tl.max((cumulative_a_amax != cumulative_a_amax).to(tl.int32), axis=1),
            axis=0,
        )
        tile_a_amax = tl.where(tile_a_has_nan != 0, float("nan"), tile_a_amax)
        tl.atomic_max(global_a_amax_ptr, tile_a_amax.to(tl.float32))

    @torch.library.custom_op("torchao::triton_rht_amax", mutates_args=())
    def triton_rht_amax(
        A: torch.Tensor,
        sign_vector: List[int],
        hadamard_dimension: int = 16,
        scaling_type: int = int(F.ScalingType.TensorWise),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RHT to A and return global absolute maxima without materializing output.

        Args:
            A: (M, N) bfloat16 tensor, row-major. M must be divisible by 16.
            sign_vector: Sign vector for the RHT as a list of ints.
            hadamard_dimension: Dimension of the Hadamard matrix (default 16).
            scaling_type: int encoding of F.ScalingType. Only TensorWise is supported.

        Returns:
            Tuple of (col_amax, row_amax):
              - col_amax: scalar float32 containing max(abs(RHT(A))).
              - row_amax: scalar float32 containing max(abs(A)).

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
        if scaling_type != int(F.ScalingType.TensorWise):
            raise ValueError(
                f"scaling_type={scaling_type!r} is not supported; "
                "only ScalingType.TensorWise is implemented."
            )
        M, N = A.shape

        sv = tuple(sign_vector)
        if hasattr(triton, "set_allocator"):
            _ws = prepare_for_cuda_graph(A.device, sign_vectors=(sv,))
            triton.set_allocator(lambda size, align, stream: _ws[: max(size, 1)])

        NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
        GROUP_SIZE_N: int = 8  # L2 reuse grouping along M

        B = get_rht_matrix(sv, A.device, torch.bfloat16, hadamard_dimension)
        global_rht_amax = torch.zeros((), dtype=torch.float32, device=A.device)
        global_a_amax = torch.zeros((), dtype=torch.float32, device=A.device)

        try:
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
        finally:
            if hasattr(triton, "set_allocator"):
                triton.set_allocator(None)
        return global_rht_amax, global_a_amax

    @triton_rht_amax.register_fake
    def _(
        A,
        sign_vector,
        hadamard_dimension=16,
        scaling_type=int(F.ScalingType.TensorWise),
    ):
        col_amax = A.new_empty((), dtype=torch.float32)
        row_amax = A.new_empty((), dtype=torch.float32)
        return col_amax, row_amax

else:

    def triton_rht_amax(
        A: torch.Tensor,
        sign_vector: list[int],
        hadamard_dimension: int = 16,
        scaling_type: int = _DEFAULT_SCALING_TYPE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_rht_amax requires torch 2.10.0+ and triton installed"
        )
