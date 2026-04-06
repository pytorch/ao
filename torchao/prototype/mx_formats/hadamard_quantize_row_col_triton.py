import torch
import torch.nn.functional as F
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

_DEFAULT_SCALING_TYPE = getattr(getattr(F, "ScalingType", None), "TensorWise", None)

if torch_version_at_least("2.10.0") and has_triton():
    import itertools

    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.hadamard_utils import (
        _compute_pid,
        _nvfp4_quantize,
        _pack_fp4,
        _store_scales_swizzle,
        _swizzle_scales,
        get_rht_matrix,
    )
    from torchao.utils import is_sm_at_least_100

    # SM100+ autotune configs. BLOCK_M=256 enables col TMA sf store; BLOCK_N=256 enables row TMA.
    HADAMARD_QUANTIZE_CONFIGS: list[triton.Config] = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_STAGES": ns},
            num_warps=nw,
            num_stages=ns,
        )
        for bm, bn, ns, nw in itertools.product(
            [128, 256],  # BLOCK_M: 256 enables columnwise TMA sf store
            [
                128,
                256,
            ],  # BLOCK_N: >= 128 for swizzle reshape; 256 enables rowwise TMA sf store
            [2, 3, 4],  # NUM_STAGES
            [4, 8],  # NUM_WARPS
        )
    ]

    @triton.autotune(
        configs=HADAMARD_QUANTIZE_CONFIGS,
        key=["M", "N", "STOCHASTIC_ROUNDING"],
    )
    @triton.jit
    def _hadamard_quantize_row_col_kernel(
        a_ptr,
        b_ptr,
        colwise_global_amax_ptr,
        rowwise_global_amax_ptr,
        colwise_c_ptr,
        colwise_sf_ptr,
        rowwise_c_ptr,
        rowwise_sf_ptr,
        col_seed_base_ptr,
        col_offset_base_ptr,
        row_seed_base_ptr,
        row_offset_base_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GROUP_SIZE_N: tl.constexpr,
        NUM_SMS: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        STOCHASTIC_ROUNDING: tl.constexpr,
    ):
        """Warp-specialized TMA kernel fusing RHT + NVFP4 columnwise quantization and
        rowwise NVFP4 quantization of the original tensor in a single pass."""
        # Create TMA descriptors in-kernel from raw pointers, shape, and stride
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
        colwise_c_desc = tl.make_tensor_descriptor(
            colwise_c_ptr,
            shape=[N, M // 2],
            strides=[M // 2, 1],
            block_shape=[BLOCK_N, BLOCK_M // 2],
        )
        # Columnwise scale factor descriptor; TMA requires contiguous dim >= 16 bytes
        # -> float8 needs BLOCK_M >= 256.
        if BLOCK_M >= 256:
            rht_col_sf_desc = tl.make_tensor_descriptor(
                colwise_sf_ptr,
                shape=[N // 128, M // 64, 32, 16],
                strides=[(M // 64) * 32 * 16, 32 * 16, 16, 1],
                block_shape=[BLOCK_N // 128, BLOCK_M // 64, 32, 16],
            )
        rowwise_c_desc = tl.make_tensor_descriptor(
            rowwise_c_ptr,
            shape=[M, N // 2],
            strides=[N // 2, 1],
            block_shape=[BLOCK_M, BLOCK_N // 2],
        )
        # TMA path: inner dim = BLOCK_N//16 bytes >= 16 bytes for BLOCK_N >= 256.
        if BLOCK_N >= 256:
            row_sf_desc = tl.make_tensor_descriptor(
                rowwise_sf_ptr,
                shape=[M // 128, N // 64, 32, 16],
                strides=[(N // 64) * 32 * 16, 32 * 16, 16, 1],
                block_shape=[BLOCK_M // 128, BLOCK_N // 64, 32, 16],
            )

        # Persistent grid-stride loop
        start_pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_N * num_pid_m
        num_tiles = num_pid_m * num_pid_n

        # Load (16, 16) random hadamard matrix once
        hadamard = b_desc.load([0, 0])

        # Load global amax scalars once
        global_amax = tl.load(colwise_global_amax_ptr)
        rowwise_global_amax = tl.load(rowwise_global_amax_ptr)

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

            # --- Columnwise path: RHT(A.t()) quantization ---

            # Transpose A_t (BLOCK_N, BLOCK_M)
            a_t = tl.trans(a)

            # Reshape to A_r (BLOCK_N * BLOCK_M // 16, 16)
            a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // 16, 16])

            # (BLOCK_N * BLOCK_M//16, 16) @ (16, 16) -> (BLOCK_N * BLOCK_M//16, 16)
            a_t_rht = tl.dot(a_t_reshape, hadamard)

            # Cast to bfloat16 like regular matmul output
            a_t_rht = a_t_rht.to(tl.bfloat16)

            # NVFP4 quantization epilogue (columnwise)
            scale_inv, scaled = _nvfp4_quantize(a_t_rht, global_amax, BLOCK_N, BLOCK_M)
            scaled_fp4x2 = _pack_fp4(
                scaled,
                BLOCK_N,
                BLOCK_M,
                STOCHASTIC_ROUNDING,
                col_seed_base_ptr,
                col_offset_base_ptr,
                tile_id,
            )

            colwise_c_desc.store([pid_n * BLOCK_N, pid_m * BLOCK_M // 2], scaled_fp4x2)

            # Store columnwise scale factors
            scale_inv = _swizzle_scales(scale_inv, BLOCK_N, BLOCK_M)
            if BLOCK_M >= 256:
                rht_col_sf_desc.store(
                    [pid_n * BLOCK_N // 128, pid_m * BLOCK_M // 64, 0, 0], scale_inv
                )
            else:
                _store_scales_swizzle(
                    scale_inv, colwise_sf_ptr, pid_n, pid_m, N, M, BLOCK_N, BLOCK_M
                )

            # --- Rowwise path: direct quantization of A (no RHT, no transpose) ---
            # a is (BLOCK_M, BLOCK_N) bfloat16, already loaded above.
            # _nvfp4_quantize treats first dim as "rows" and second as inner (M//16 vectors).
            # Calling with (BLOCK_M, BLOCK_N) quantizes each row of A in blocks of 16 along N.
            rowwise_scale_inv, rowwise_scaled = _nvfp4_quantize(
                a, rowwise_global_amax, BLOCK_M, BLOCK_N
            )
            rowwise_fp4x2 = _pack_fp4(
                rowwise_scaled,
                BLOCK_M,
                BLOCK_N,
                STOCHASTIC_ROUNDING,
                row_seed_base_ptr,
                row_offset_base_ptr,
                tile_id,
            )

            rowwise_c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N // 2], rowwise_fp4x2)

            rowwise_scale_inv = _swizzle_scales(rowwise_scale_inv, BLOCK_M, BLOCK_N)
            if BLOCK_N >= 256:
                row_sf_desc.store(
                    [pid_m * BLOCK_M // 128, pid_n * BLOCK_N // 64, 0, 0],
                    rowwise_scale_inv,
                )
            else:
                _store_scales_swizzle(
                    rowwise_scale_inv,
                    rowwise_sf_ptr,
                    pid_m,
                    pid_n,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                )

    def triton_rht_quantize_row_col(
        A: torch.Tensor,
        col_global_amax: torch.Tensor,
        row_global_amax: torch.Tensor,
        stochastic_rounding: bool = False,
        sign_vector: tuple[int, ...] | None = None,
        hadamard_dimension: int = 16,
        scaling_type: F.ScalingType = F.ScalingType.TensorWise,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """RHT + NVFP4 E2M1 columnwise quantization fused with rowwise quantization.

        Produces both:
          - Columnwise output: quantization of RHT(A.t()), shape (N, M//2) +
            (N//128, M//64, 32, 16) swizzled scales.
          - Rowwise output: direct NVFP4 quantization of A, shape (M, N//2) +
            (M//128, N//64, 32, 16) swizzled scales.

        Both paths share the same TMA tile loads. The caller computes and passes global
        amax values so they can be reused or all-reduced before quantization.

        Args:
            A: (M, N) bfloat16 tensor, row-major. M and N must be divisible by 128.
            col_global_amax: scalar float32 global amax of RHT(A.t.).
            row_global_amax: scalar float32 global amax of A.
            stochastic_rounding: Use stochastic rounding for both columnwise and rowwise FP4 paths.
            sign_vector: Optional sign vector for the RHT. If None, a random one is generated.
            hadamard_dimension: Dimension of the Hadamard matrix (default 16).
            scaling_type: ScalingType controlling reduction granularity. Only
                ``ScalingType.TensorWise`` is currently supported.

        Returns:
            Tuple of (col_fp4, col_sf, row_fp4, row_sf):
              - col_fp4:  (N, M//2) uint8 packed FP4 codes (columnwise).
              - col_sf:   (N//128, M//64, 32, 16) swizzled float8_e4m3fn scale factors.
              - row_fp4:  (M, N//2) uint8 packed FP4 codes (rowwise).
              - row_sf:   (M//128, N//64, 32, 16) swizzled float8_e4m3fn scale factors.

        Raises:
            NotImplementedError: If hardware is pre-SM100.
            ValueError: If A is not bfloat16, not 2-D, not contiguous, M/N not divisible
                by 128, scaling_type is not ScalingType.TensorWise, or global amax
                tensors are not single-element float32 tensors.
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
        if A.shape[0] % 128 != 0:
            raise ValueError(
                f"M must be divisible by 128 for swizzled scales, got M={A.shape[0]}"
            )
        if scaling_type != F.ScalingType.TensorWise:
            raise ValueError(
                f"scaling_type={scaling_type!r} is not supported; "
                "only ScalingType.TensorWise is implemented."
            )
        M, N = A.shape
        if N % 128 != 0:
            raise ValueError(
                f"N must be divisible by 128 for swizzled scales, got N={N}"
            )
        if col_global_amax.numel() != 1:
            raise ValueError(
                f"col_global_amax must contain a single element, got {col_global_amax.numel()}"
            )
        if col_global_amax.dtype != torch.float32:
            raise ValueError(
                f"col_global_amax must be float32, got {col_global_amax.dtype}"
            )
        if row_global_amax.numel() != 1:
            raise ValueError(
                f"row_global_amax must contain a single element, got {row_global_amax.numel()}"
            )
        if row_global_amax.dtype != torch.float32:
            raise ValueError(
                f"row_global_amax must be float32, got {row_global_amax.dtype}"
            )

        if stochastic_rounding:
            col_seed_base = torch.randint(
                low=-(2**63),
                high=2**63 - 1,
                size=(1,),
                dtype=torch.int64,
                device=A.device,
            )
            col_offset_base = torch.randint(
                low=-(2**63),
                high=2**63 - 1,
                size=(1,),
                dtype=torch.int64,
                device=A.device,
            )
            row_seed_base = torch.randint(
                low=-(2**63),
                high=2**63 - 1,
                size=(1,),
                dtype=torch.int64,
                device=A.device,
            )
            row_offset_base = torch.randint(
                low=-(2**63),
                high=2**63 - 1,
                size=(1,),
                dtype=torch.int64,
                device=A.device,
            )
        else:
            col_seed_base = 0  # Safe NULL value for Triton
            col_offset_base = 0
            row_seed_base = 0
            row_offset_base = 0

        NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
        GROUP_SIZE_N: int = 8

        B = get_rht_matrix(
            sign_vector=sign_vector,
            device=A.device,
            hadamard_dimension=hadamard_dimension,
        ).to(torch.bfloat16)

        # Columnwise outputs
        colwise_C = torch.empty((N, M // 2), dtype=torch.uint8, device=A.device)
        colwise_sf = torch.empty(
            (N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )

        # Rowwise outputs
        rowwise_C = torch.empty((M, N // 2), dtype=torch.uint8, device=A.device)
        rowwise_sf = torch.empty(
            (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )

        # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch space.
        # Outside torch.compile, none is set by default; mirror what torch._inductor does.
        if hasattr(triton, "set_allocator"):
            triton.set_allocator(
                lambda size, align, stream: torch.empty(
                    size, dtype=torch.int8, device=A.device
                )
            )

        _hadamard_quantize_row_col_kernel[(NUM_SMS,)](
            A,
            B,
            col_global_amax,
            row_global_amax,
            colwise_C,
            colwise_sf,
            rowwise_C,
            rowwise_sf,
            col_seed_base,
            col_offset_base,
            row_seed_base,
            row_offset_base,
            M,
            N,
            GROUP_SIZE_N=GROUP_SIZE_N,
            NUM_SMS=NUM_SMS,
            STOCHASTIC_ROUNDING=stochastic_rounding,
        )

        return colwise_C, colwise_sf, rowwise_C, rowwise_sf

else:

    def triton_rht_quantize_row_col(
        A: torch.Tensor,
        col_global_amax: torch.Tensor,
        row_global_amax: torch.Tensor,
        stochastic_rounding: bool = False,
        sign_vector: tuple[int, ...] | None = None,
        hadamard_dimension: int = 16,
        scaling_type: object = _DEFAULT_SCALING_TYPE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_rht_quantize_row_col requires torch 2.10.0+ and triton installed"
        )
