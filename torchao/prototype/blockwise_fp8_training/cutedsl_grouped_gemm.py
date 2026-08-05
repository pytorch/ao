# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import weakref
from typing import Tuple

import torch

from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _cutedsl_runtime_available,
    _missing_cutedsl_runtime_packages,
)
from torchao.utils import ceil_div

_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM_ENV = (
    "TORCHAO_ENABLE_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM"
)
_HOPPER_DENSE_GEMM_COMPILED = {}
_HOPPER_DENSE_PERSISTENT_GEMM_COMPILED = {}
_HOPPER_DENSE_KBLOCK_BATCHED_GEMM_COMPILED = {}
_HOPPER_DENSE_PERSISTENT_KBLOCK_BATCHED_GEMM_COMPILED = {}
_HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED = {}
_HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE = {}
_HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE = {}
_HOPPER_DENSE_B_SCALE_TENSOR_CACHE = {}
_HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE_MAX_ENTRIES = 16
_SPLITK_BF16_WORKSPACE_CACHE = {}
_SPLITK_BF16_SPLIT_PARTIAL_WORKSPACE_CACHE = {}
_SPLITK_BF16_WORKSPACE_CACHE_MAX_ENTRIES = 4
_WGRAD_BF16_PARTIAL_WORKSPACE_CACHE = {}
_WGRAD_BF16_PARTIAL_WORKSPACE_CACHE_MAX_ENTRIES = 2
_CUDA_STREAM_CACHE = {}
_HOPPER_DENSE_TILE_SHAPE_MN = (128, 128)
_HOPPER_DENSE_WIDE_N_TILE_SHAPE_MN = (128, 256)
_HOPPER_DENSE_FWD_TILE_SHAPE_MN = (128, 192)
_EQUAL_GROUP_OFFSETS_CACHE = {}
_WGRAD_OFFSETS_VALIDATION_CACHE = {}
_M_GROUPED_LAYOUT_CACHE = {}
_M_GROUPED_LAYOUT_CACHE_MAX_ENTRIES = 16
_SCALE_ACCUM_MAX_CHUNK_BLOCKS = 64
_SCALE_ACCUM_MAX_PARTIAL_BYTES = 8 * 1024 * 1024 * 1024
_SCALE_ACCUM_HUGE_WIDE_SPLIT8_PARTIAL_BYTES = 16 * 1024 * 1024 * 1024
_SCALE_OUTPUT_MAX_BF16_PARTIAL_BYTES = 2 * 1024 * 1024 * 1024
_SCALE_OUTPUT_SPLIT16_MAX_BF16_PARTIAL_BYTES = 32 * 1024 * 1024 * 1024


def _cutedsl_fp8_blockwise_grouped_mm_enabled() -> bool:
    # Keep the prototype opt-in while the CuTeDSL kernels are experimental.
    return os.environ.get(_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM_ENV, "0") == "1"


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _cutlass_dtype(dtype_name: str):
    import cutlass

    if dtype_name == "float8_e4m3fn":
        return cutlass.Float8E4M3FN
    if dtype_name == "bfloat16":
        return cutlass.BFloat16
    if dtype_name == "float32":
        return cutlass.Float32
    raise NotImplementedError(f"unsupported CuTeDSL dtype: torch.{dtype_name}")


@functools.cache
def _load_cutedsl_hopper_gemm_module():
    if not _cutedsl_runtime_available():
        return None

    from torchao.prototype.blockwise_fp8_training import _cutedsl_hopper_gemm

    return _cutedsl_hopper_gemm


def _load_hopper_dense_gemm_module():
    return _load_cutedsl_hopper_gemm_module()


def _load_hopper_dense_gemm_persistent_module():
    return _load_cutedsl_hopper_gemm_module()


def _hopper_dense_gemm_available() -> bool:
    return _cutedsl_runtime_available()


def _is_cutedsl_2d_3d_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> bool:
    # This predicate covers the 2D x 3D grouped GEMM path used by forward and
    # dgrad; wgrad has a separate 2D x 2D predicate below.
    return (
        _cutedsl_fp8_blockwise_grouped_mm_enabled()
        and _cutedsl_runtime_available()
        and a.is_cuda
        and b.is_cuda
        and a.ndim == 2
        and b.ndim == 3
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == torch.float8_e4m3fn
        and a_s.dtype == torch.float32
        and b_s.dtype == torch.float32
        and offs.dtype == torch.int32
        and scale_recipe_a == 4
        and scale_recipe_b == 5
        and out_dtype in (torch.bfloat16, torch.float32)
        and block_size == 128
    )


def _is_cutedsl_2d_2d_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> bool:
    if not (
        _cutedsl_fp8_blockwise_grouped_mm_enabled()
        and _cutedsl_runtime_available()
        and a.is_cuda
        and b.is_cuda
        and a.ndim == 2
        and b.ndim == 2
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == torch.float8_e4m3fn
        and a_s.dtype == torch.float32
        and b_s.dtype == torch.float32
        and offs.dtype == torch.int32
        and scale_recipe_a == 4
        and scale_recipe_b == 4
        and out_dtype in (torch.bfloat16, torch.float32)
        and block_size == 128
        and a.shape[1] == b.shape[0]
        and a.shape[1] % block_size == 0
    ):
        return False
    return _wgrad_offsets_are_valid(offs, a.shape[1], block_size)


def _can_use_cutedsl_fp8_blockwise_grouped_mm_training(
    a: torch.Tensor,
    b_t: torch.Tensor,
    group_end_offsets: torch.Tensor,
    out_dtype: torch.dtype,
    float8_dtype: torch.dtype,
    block_size: int,
    num_rows: int,
) -> bool:
    if not (
        _cutedsl_fp8_blockwise_grouped_mm_enabled()
        and _cutedsl_runtime_available()
        and torch.cuda.is_available()
        and not torch.version.hip
        and a.is_cuda
        and b_t.is_cuda
        and group_end_offsets.is_cuda
        and a.device == b_t.device == group_end_offsets.device
        and a.ndim == 2
        and b_t.ndim == 3
        and group_end_offsets.ndim == 1
        and group_end_offsets.dtype == torch.int32
        and a.dtype in (torch.bfloat16, torch.float32)
        and b_t.dtype in (torch.bfloat16, torch.float32)
        and float8_dtype == torch.float8_e4m3fn
        and a.stride(-1) == 1
        and b_t.stride(-2) == 1
        and a.shape[-1] == b_t.shape[-2]
        and b_t.shape[0] == group_end_offsets.numel()
        and num_rows > 0
        and a.shape[-1] > 0
        and b_t.shape[-1] > 0
        and a.shape[-1] % block_size == 0
        and b_t.shape[-1] % block_size == 0
        and out_dtype in (torch.bfloat16, torch.float32)
        and block_size == 128
    ):
        return False
    major, _ = torch.cuda.get_device_capability(a.device)
    return major >= 9 and _wgrad_offsets_are_valid(
        group_end_offsets, num_rows, block_size
    )


@functools.cache
def _compile_fp8_blockwise_grouped_gemm_2d_3d(
    a_dtype_name: str,
    b_dtype_name: str,
    c_dtype_name: str,
    block_size: int,
    threads_per_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    A_DTYPE = _cutlass_dtype(a_dtype_name)
    B_DTYPE = _cutlass_dtype(b_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)

    class Fp8BlockwiseGroupedGemm2d3dKernel:
        @cute.kernel
        def kernel(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            grouped_layout: cute.Tensor,
            c: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            K: cutlass.Int64,
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m = cutlass.Int64(m_tile)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)

            if m < M and n < N:
                group = cutlass.Int64(grouped_layout[m])
                n_block = n // BLOCK_SIZE
                acc = cutlass.Float32(0.0)
                k4_extent = cutlass.Int32(K // 4)
                a_u32 = cute.recast_tensor(a, cutlass.Uint32)
                b_u32 = cute.recast_tensor(b, cutlass.Uint32)

                for k4_i in cutlass.range(0, k4_extent, 1, unroll=1):
                    k = cutlass.Int64(k4_i * 4)
                    k_block = k // BLOCK_SIZE
                    a_scale = cutlass.Float32(a_s[m, k_block])
                    b_scale = cutlass.Float32(b_s[group, k_block, n_block])

                    a_vals = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
                    b_vals = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
                    a_vals_u32 = cute.recast_tensor(a_vals, cutlass.Uint32)
                    b_vals_u32 = cute.recast_tensor(b_vals, cutlass.Uint32)
                    a_vals_u32[0] = a_u32[m, k4_i]
                    b_vals_u32[0] = b_u32[group, k4_i, n]
                    a_f32 = a_vals.load().to(cutlass.Float32) * a_scale
                    b_f32 = b_vals.load().to(cutlass.Float32) * b_scale
                    acc += (a_f32 * b_f32).reduce(
                        cute.ReductionOp.ADD, cutlass.Float32(0.0), 0
                    )

                c[m, n] = C_DTYPE(acc)

        @cute.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            grouped_layout: cute.Tensor,
            c: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            K: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                a,
                b,
                a_s,
                b_s,
                grouped_layout,
                c,
                M,
                N,
                K,
                BLOCK_SIZE=block_size,
                THREADS_PER_BLOCK=threads_per_block,
            ).launch(
                grid=(cute.ceil_div(N, threads_per_block) * M, 1, 1),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    kernel = Fp8BlockwiseGroupedGemm2d3dKernel()

    m = cute.sym_int()
    e = cute.sym_int()
    k = cute.sym_int(divisibility=block_size)
    n = cute.sym_int(divisibility=block_size)
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    a_stride0 = cute.sym_int()
    a_stride1 = cute.sym_int()
    b_stride0 = cute.sym_int()
    b_stride1 = cute.sym_int()
    b_stride2 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    c_stride0 = cute.sym_int()
    c_stride1 = cute.sym_int()
    grouped_layout_stride0 = cute.sym_int()

    fake_a = make_fake_tensor(
        A_DTYPE,
        (m, k),
        stride=(a_stride0, a_stride1),
    )
    fake_b = make_fake_tensor(
        B_DTYPE,
        (e, k, n),
        stride=(b_stride0, b_stride1, b_stride2),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_grouped_layout = make_fake_tensor(
        cutlass.Int32,
        (m,),
        stride=(grouped_layout_stride0,),
    )
    fake_c = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(c_stride0, c_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        a=fake_a,
        b=fake_b,
        a_s=fake_a_s,
        b_s=fake_b_s,
        grouped_layout=fake_grouped_layout,
        c=fake_c,
        M=0,
        N=0,
        K=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_fp8_blockwise_grouped_gemm_2d_2d(
    a_dtype_name: str,
    b_dtype_name: str,
    c_dtype_name: str,
    block_size: int,
    threads_per_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    A_DTYPE = _cutlass_dtype(a_dtype_name)
    B_DTYPE = _cutlass_dtype(b_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)

    class Fp8BlockwiseGroupedGemm2d2dKernel:
        @cute.kernel
        def kernel(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            offs: cute.Tensor,
            c: cute.Tensor,
            E: cutlass.Int64,
            N: cutlass.Int64,
            K: cutlass.Int64,
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            k_tiles = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            tiles_per_expert = N * k_tiles
            expert = cta_idx // tiles_per_expert
            expert_tile = cta_idx - expert * tiles_per_expert
            n = expert_tile // k_tiles
            k_tile = expert_tile - n * k_tiles
            k = cutlass.Int64(k_tile * THREADS_PER_BLOCK + tidx)

            if expert < E and n < N and k < K:
                group_start = cutlass.Int64(0)
                if expert > 0:
                    group_start = cutlass.Int64(offs[expert - 1])
                group_end = cutlass.Int64(offs[expert])
                first_scale_block = cutlass.Int32(group_start // BLOCK_SIZE)
                last_scale_block = cutlass.Int32(group_end // BLOCK_SIZE)
                a_u32 = cute.recast_tensor(a, cutlass.Uint32)
                b_u32 = cute.recast_tensor(b, cutlass.Uint32)
                acc = cutlass.Float32(0.0)

                for scale_block in cutlass.range(
                    first_scale_block,
                    last_scale_block,
                    1,
                    unroll=1,
                ):
                    partial = cutlass.Float32(0.0)
                    k4_start = cutlass.Int32(scale_block * (BLOCK_SIZE // 4))
                    for k4_offset in cutlass.range_constexpr(BLOCK_SIZE // 4):
                        k4 = k4_start + k4_offset
                        a_vals = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
                        b_vals = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
                        a_vals_u32 = cute.recast_tensor(a_vals, cutlass.Uint32)
                        b_vals_u32 = cute.recast_tensor(b_vals, cutlass.Uint32)
                        a_vals_u32[0] = a_u32[n, k4]
                        b_vals_u32[0] = b_u32[k4, k]
                        a_f32 = a_vals.load().to(cutlass.Float32)
                        b_f32 = b_vals.load().to(cutlass.Float32)
                        partial += (a_f32 * b_f32).reduce(
                            cute.ReductionOp.ADD,
                            cutlass.Float32(0.0),
                            0,
                        )
                    acc += (
                        partial
                        * cutlass.Float32(a_s[n, scale_block])
                        * cutlass.Float32(b_s[scale_block, k])
                    )
                c[expert, n, k] = C_DTYPE(acc)

        @cute.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            offs: cute.Tensor,
            c: cute.Tensor,
            E: cutlass.Int64,
            N: cutlass.Int64,
            K: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                a,
                b,
                a_s,
                b_s,
                offs,
                c,
                E,
                N,
                K,
                BLOCK_SIZE=block_size,
                THREADS_PER_BLOCK=threads_per_block,
            ).launch(
                grid=(E * N * cute.ceil_div(K, threads_per_block), 1, 1),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    kernel = Fp8BlockwiseGroupedGemm2d2dKernel()
    e = cute.sym_int()
    n = cute.sym_int()
    m = cute.sym_int(divisibility=block_size)
    k = cute.sym_int()
    m_blocks = cute.sym_int()
    a_stride0 = cute.sym_int()
    a_stride1 = cute.sym_int()
    b_stride0 = cute.sym_int()
    b_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    c_stride0 = cute.sym_int()
    c_stride1 = cute.sym_int()
    c_stride2 = cute.sym_int()
    offs_stride0 = cute.sym_int()

    fake_a = make_fake_tensor(A_DTYPE, (n, m), stride=(a_stride0, a_stride1))
    fake_b = make_fake_tensor(B_DTYPE, (m, k), stride=(b_stride0, b_stride1))
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (n, m_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (m_blocks, k),
        stride=(b_s_stride0, b_s_stride1),
    )
    fake_offs = make_fake_tensor(cutlass.Int32, (e,), stride=(offs_stride0,))
    fake_c = make_fake_tensor(
        C_DTYPE,
        (e, n, k),
        stride=(c_stride0, c_stride1, c_stride2),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        a=fake_a,
        b=fake_b,
        a_s=fake_a_s,
        b_s=fake_b_s,
        offs=fake_offs,
        c=fake_c,
        E=0,
        N=0,
        K=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def _equal_group_size_from_offsets(offs: torch.Tensor, M: int) -> int | None:
    E = offs.numel()
    if E == 0 or M % E != 0:
        return None

    version = getattr(offs, "_version", None)
    cache_key = (
        offs.data_ptr(),
        tuple(offs.shape),
        tuple(offs.stride()),
        offs.storage_offset(),
        offs.dtype,
        offs.device,
        M,
        version,
    )
    cached = _EQUAL_GROUP_OFFSETS_CACHE.get(cache_key)
    if cached is not None:
        cached_offs, cached_m_per_group = cached
        if cached_offs() is offs:
            return cached_m_per_group

    M_per_group = M // E
    expected = torch.arange(
        M_per_group,
        (E + 1) * M_per_group,
        M_per_group,
        device=offs.device,
        dtype=offs.dtype,
    )
    if not torch.equal(offs, expected):
        _EQUAL_GROUP_OFFSETS_CACHE[cache_key] = (weakref.ref(offs), None)
        return None
    _EQUAL_GROUP_OFFSETS_CACHE[cache_key] = (weakref.ref(offs), M_per_group)
    return M_per_group


def _wgrad_offsets_are_valid(
    offs: torch.Tensor,
    reduction_extent: int,
    block_size: int,
) -> bool:
    version = getattr(offs, "_version", None)
    cache_key = (
        offs.data_ptr(),
        tuple(offs.shape),
        tuple(offs.stride()),
        offs.storage_offset(),
        offs.dtype,
        offs.device,
        reduction_extent,
        block_size,
        version,
    )
    cached = _WGRAD_OFFSETS_VALIDATION_CACHE.get(cache_key)
    if cached is not None:
        cached_offs, valid = cached
        if cached_offs() is offs:
            return valid

    values = offs.detach().cpu().tolist()
    valid = bool(values)
    previous = 0
    for value in values:
        valid = valid and previous <= value <= reduction_extent
        valid = valid and value % block_size == 0
        previous = value

    if len(_WGRAD_OFFSETS_VALIDATION_CACHE) >= 16:
        _WGRAD_OFFSETS_VALIDATION_CACHE.pop(
            next(iter(_WGRAD_OFFSETS_VALIDATION_CACHE)),
            None,
        )
    _WGRAD_OFFSETS_VALIDATION_CACHE[cache_key] = (weakref.ref(offs), valid)
    return valid


def _m_grouped_layout_from_offsets(offs: torch.Tensor, M: int) -> torch.Tensor:
    """Build the row-to-expert map used by contiguous M-grouped kernels."""
    version = getattr(offs, "_version", None)
    cache_key = (
        offs.data_ptr(),
        tuple(offs.shape),
        tuple(offs.stride()),
        offs.storage_offset(),
        offs.dtype,
        offs.device,
        M,
        version,
    )
    cached = _M_GROUPED_LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        cached_offs, grouped_layout = cached
        if cached_offs() is offs:
            return grouped_layout

    rows = torch.arange(M, dtype=torch.int32, device=offs.device)
    grouped_layout = torch.bucketize(rows, offs, right=True, out_int32=True).clamp_max(
        offs.numel() - 1
    )
    if len(_M_GROUPED_LAYOUT_CACHE) >= _M_GROUPED_LAYOUT_CACHE_MAX_ENTRIES:
        _M_GROUPED_LAYOUT_CACHE.pop(next(iter(_M_GROUPED_LAYOUT_CACHE)))
    _M_GROUPED_LAYOUT_CACHE[cache_key] = (weakref.ref(offs), grouped_layout)
    return grouped_layout


def _make_cutedsl_tensor(
    torch_tensor: torch.Tensor,
    cutlass_dtype,
    leading_dim: int | None,
):
    from cutlass.cute.runtime import from_dlpack

    cute_tensor = from_dlpack(torch_tensor, assumed_align=16, enable_tvm_ffi=True)
    cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    cute_tensor.element_type = cutlass_dtype
    return cute_tensor


def _make_hopper_dense_tensor(torch_tensor: torch.Tensor, cutlass_dtype):
    return _make_cutedsl_tensor(torch_tensor, cutlass_dtype, leading_dim=1)


def _get_cuda_stream(stream_id: int):
    stream = _CUDA_STREAM_CACHE.get(stream_id)
    if stream is None:
        import cuda.bindings.driver as cuda

        stream = cuda.CUstream(stream_id)
        _CUDA_STREAM_CACHE[stream_id] = stream
    return stream


def _get_cached_hopper_dense_b_kblock_tensors(
    b: torch.Tensor,
    block_size: int,
    cutlass_dtype,
):
    E, K, N = b.shape
    if K % block_size != 0:
        return None

    cache_key = (
        id(b),
        b.data_ptr(),
        tuple(b.shape),
        tuple(b.stride()),
        b.storage_offset(),
        b.dtype,
        b.device,
        block_size,
    )
    cached = _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        cached_b, cached_tensors = cached
        if cached_b() is b:
            return cached_tensors

    tensors = []
    for k_block in range(K // block_size):
        b_view = torch.as_strided(
            b,
            (N, block_size, E),
            (K, 1, K * N),
            storage_offset=k_block * block_size,
        )
        tensors.append(_make_hopper_dense_tensor(b_view, cutlass_dtype))

    if len(_HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE) >= (
        _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE_MAX_ENTRIES
    ):
        for key, (cached_b, _) in list(_HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE.items()):
            if cached_b() is None:
                _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE.pop(key, None)
                break
        else:
            _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE.pop(
                next(iter(_HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE)),
                None,
            )
    _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE[cache_key] = (weakref.ref(b), tensors)
    return tensors


def _get_cached_hopper_dense_b_kblock_batched_tensor(
    b: torch.Tensor,
    block_size: int,
    chunk_start: int,
    chunk_blocks: int,
    cutlass_dtype,
    make_view: bool,
):
    E, K, N = b.shape
    storage_offset = chunk_start * block_size
    cache_key = (
        id(b),
        b.data_ptr(),
        tuple(b.shape),
        tuple(b.stride()),
        b.storage_offset(),
        b.dtype,
        b.device,
        block_size,
        chunk_start,
        chunk_blocks,
    )
    cached = _HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        cached_b, cached_tensor = cached
        if cached_b() is b:
            if not make_view:
                return None, cached_tensor
            b_view = torch.as_strided(
                b,
                (N, block_size, E, chunk_blocks),
                (K, 1, K * N, block_size),
                storage_offset=storage_offset,
            )
            return b_view, cached_tensor

    b_view = torch.as_strided(
        b,
        (N, block_size, E, chunk_blocks),
        (K, 1, K * N, block_size),
        storage_offset=storage_offset,
    )
    tensor = _make_hopper_dense_tensor(b_view, cutlass_dtype)
    if len(_HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE) >= (
        _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE_MAX_ENTRIES
    ):
        for key, (cached_b, _) in list(
            _HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE.items()
        ):
            if cached_b() is None:
                _HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE.pop(key, None)
                break
        else:
            _HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE.pop(
                next(iter(_HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE)),
                None,
            )
    _HOPPER_DENSE_B_KBLOCK_BATCHED_TENSOR_CACHE[cache_key] = (
        weakref.ref(b),
        tensor,
    )
    return b_view, tensor


def _get_cached_b_scale_tensor(b_s: torch.Tensor, cutlass_dtype):
    cache_key = (
        id(b_s),
        b_s.data_ptr(),
        tuple(b_s.shape),
        tuple(b_s.stride()),
        b_s.storage_offset(),
        b_s.dtype,
        b_s.device,
    )
    cached = _HOPPER_DENSE_B_SCALE_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        cached_b_s, cached_tensor = cached
        if cached_b_s() is b_s:
            return cached_tensor

    tensor = _make_cutedsl_tensor(b_s, cutlass_dtype, leading_dim=1)
    if len(_HOPPER_DENSE_B_SCALE_TENSOR_CACHE) >= (
        _HOPPER_DENSE_B_KBLOCK_TENSOR_CACHE_MAX_ENTRIES
    ):
        for key, (cached_b_s, _) in list(_HOPPER_DENSE_B_SCALE_TENSOR_CACHE.items()):
            if cached_b_s() is None:
                _HOPPER_DENSE_B_SCALE_TENSOR_CACHE.pop(key, None)
                break
        else:
            _HOPPER_DENSE_B_SCALE_TENSOR_CACHE.pop(
                next(iter(_HOPPER_DENSE_B_SCALE_TENSOR_CACHE)),
                None,
            )
    _HOPPER_DENSE_B_SCALE_TENSOR_CACHE[cache_key] = (weakref.ref(b_s), tensor)
    return tensor


def _get_cached_splitk_bf16_workspace(
    M: int,
    N: int,
    E: int,
    M_per_group: int,
    scale_chunk_blocks: int,
    needs_accum: bool,
    accum_dtype: torch.dtype,
    partial_dtype: torch.dtype,
    device: torch.device,
    stream_id: int,
    cutlass_partial_dtype,
    make_batched_c_tensor: bool,
    make_scale_partials_tensor: bool,
):
    cache_key = (
        device,
        stream_id,
        M,
        N,
        E,
        M_per_group,
        scale_chunk_blocks,
        needs_accum,
        accum_dtype,
        partial_dtype,
        make_batched_c_tensor,
        make_scale_partials_tensor,
    )
    cached = _SPLITK_BF16_WORKSPACE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    accum = (
        torch.empty((M, N), device=device, dtype=accum_dtype) if needs_accum else None
    )
    partials = torch.empty(
        (scale_chunk_blocks, M, N), device=device, dtype=partial_dtype
    )
    c_views = [
        torch.as_strided(
            partials[local_k],
            (M_per_group, N, E),
            (N, 1, M_per_group * N),
        )
        for local_k in range(scale_chunk_blocks)
    ]
    mCs = [
        _make_hopper_dense_tensor(c_view, cutlass_partial_dtype) for c_view in c_views
    ]
    batched_c_view = None
    batched_mC = None
    if make_batched_c_tensor:
        batched_c_view = torch.as_strided(
            partials,
            (M_per_group, N, E, scale_chunk_blocks),
            (N, 1, M_per_group * N, M * N),
        )
        batched_mC = _make_hopper_dense_tensor(batched_c_view, cutlass_partial_dtype)
    scale_partials = (
        _make_cutedsl_tensor(partials, cutlass_partial_dtype, leading_dim=2)
        if make_scale_partials_tensor
        else None
    )

    if len(_SPLITK_BF16_WORKSPACE_CACHE) >= _SPLITK_BF16_WORKSPACE_CACHE_MAX_ENTRIES:
        _SPLITK_BF16_WORKSPACE_CACHE.pop(
            next(iter(_SPLITK_BF16_WORKSPACE_CACHE)),
            None,
        )
    workspace = (
        partials,
        accum,
        c_views,
        mCs,
        batched_c_view,
        batched_mC,
        scale_partials,
    )
    _SPLITK_BF16_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def _get_cached_splitk_bf16_split_partial_workspace(
    M: int,
    N: int,
    E: int,
    M_per_group: int,
    scale_chunk_blocks: int,
    needs_accum: bool,
    accum_dtype: torch.dtype,
    device: torch.device,
    stream_id: int,
    cutlass_partial_dtype,
):
    cache_key = (
        device,
        stream_id,
        M,
        N,
        E,
        M_per_group,
        scale_chunk_blocks,
        needs_accum,
        accum_dtype,
    )
    cached = _SPLITK_BF16_SPLIT_PARTIAL_WORKSPACE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    accum = (
        torch.empty((M, N), device=device, dtype=accum_dtype) if needs_accum else None
    )
    partials = [
        torch.empty((M, N), device=device, dtype=torch.bfloat16)
        for _ in range(scale_chunk_blocks)
    ]
    c_views = [
        torch.as_strided(
            partial,
            (M_per_group, N, E),
            (N, 1, M_per_group * N),
        )
        for partial in partials
    ]
    mCs = [
        _make_hopper_dense_tensor(c_view, cutlass_partial_dtype) for c_view in c_views
    ]

    if (
        len(_SPLITK_BF16_SPLIT_PARTIAL_WORKSPACE_CACHE)
        >= _SPLITK_BF16_WORKSPACE_CACHE_MAX_ENTRIES
    ):
        _SPLITK_BF16_SPLIT_PARTIAL_WORKSPACE_CACHE.pop(
            next(iter(_SPLITK_BF16_SPLIT_PARTIAL_WORKSPACE_CACHE)),
            None,
        )
    workspace = (partials, accum, c_views, mCs)
    _SPLITK_BF16_SPLIT_PARTIAL_WORKSPACE_CACHE[cache_key] = workspace
    return workspace


def _get_hopper_dense_kblock_gemm(
    a_view: torch.Tensor,
    b_view: torch.Tensor,
    c_view: torch.Tensor,
    key: tuple,
    tile_shape_mn: tuple[int, int],
    c_cutlass_dtype,
):
    compiled = _HOPPER_DENSE_GEMM_COMPILED.get(key)
    if compiled is not None:
        return compiled

    module = _load_hopper_dense_gemm_module()
    if module is None:
        return None

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
    mB = _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN)
    mC = _make_hopper_dense_tensor(c_view, c_cutlass_dtype)
    gemm = module.HopperWgmmaGemmKernel(cutlass.Float32, tile_shape_mn, (1, 1))
    compiled = cute.compile(gemm, mA, mB, mC, stream)
    _HOPPER_DENSE_GEMM_COMPILED[key] = compiled
    return compiled


def _get_hopper_dense_persistent_kblock_gemm(
    a_view: torch.Tensor,
    b_view: torch.Tensor,
    c_view: torch.Tensor,
    key: tuple,
    tile_shape_mn: tuple[int, int],
    c_cutlass_dtype,
):
    compiled = _HOPPER_DENSE_PERSISTENT_GEMM_COMPILED.get(key)
    if compiled is not None:
        return compiled

    module = _load_hopper_dense_gemm_persistent_module()
    if module is None:
        return None

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
    mB = _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN)
    mC = _make_hopper_dense_tensor(c_view, c_cutlass_dtype)
    cluster_shape_mn = _hopper_dense_persistent_cluster_shape_mn(tile_shape_mn)
    gemm = module.HopperWgmmaGemmPersistentKernel(
        cutlass.Float32,
        tile_shape_mn,
        cluster_shape_mn,
        1,
        False,
    )
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    compiled = cute.compile(gemm, mA, mB, mC, max_active_clusters, stream)
    _HOPPER_DENSE_PERSISTENT_GEMM_COMPILED[key] = compiled
    return compiled


def _get_hopper_dense_kblock_batched_gemm(
    a_view: torch.Tensor,
    b_view: torch.Tensor,
    c_view: torch.Tensor,
    key: tuple,
    tile_shape_mn: tuple[int, int],
    c_cutlass_dtype,
):
    compiled = _HOPPER_DENSE_KBLOCK_BATCHED_GEMM_COMPILED.get(key)
    if compiled is not None:
        return compiled

    module = _load_hopper_dense_gemm_module()
    if module is None:
        return None

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    class HopperWgmmaKBlockBatchedGemmKernel:
        def __init__(self):
            self.gemm = module.HopperWgmmaGemmKernel(
                cutlass.Float32,
                tile_shape_mn,
                (1, 1),
            )

        @cute.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            c: cute.Tensor,
            stream: cuda.CUstream,
        ):
            a = cute.group_modes(a, 2, 4)
            b = cute.group_modes(b, 2, 4)
            c = cute.group_modes(c, 2, 4)
            self.gemm(a, b, c, stream)

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
    mB = _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN)
    mC = _make_hopper_dense_tensor(c_view, c_cutlass_dtype)
    compiled = cute.compile(
        HopperWgmmaKBlockBatchedGemmKernel(),
        mA,
        mB,
        mC,
        stream,
    )
    _HOPPER_DENSE_KBLOCK_BATCHED_GEMM_COMPILED[key] = compiled
    return compiled


def _get_hopper_dense_persistent_kblock_batched_gemm(
    a_view: torch.Tensor,
    b_view: torch.Tensor,
    c_view: torch.Tensor,
    key: tuple,
    tile_shape_mn: tuple[int, int],
    c_cutlass_dtype,
):
    compiled = _HOPPER_DENSE_PERSISTENT_KBLOCK_BATCHED_GEMM_COMPILED.get(key)
    if compiled is not None:
        return compiled

    module = _load_hopper_dense_gemm_persistent_module()
    if module is None:
        return None

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    class HopperWgmmaKBlockBatchedPersistentGemmKernel:
        def __init__(self):
            cluster_shape_mn = _hopper_dense_persistent_cluster_shape_mn(tile_shape_mn)
            self.gemm = module.HopperWgmmaGemmPersistentKernel(
                cutlass.Float32,
                tile_shape_mn,
                cluster_shape_mn,
                1,
                False,
            )

        @cute.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            c: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            a = cute.group_modes(a, 2, 4)
            b = cute.group_modes(b, 2, 4)
            c = cute.group_modes(c, 2, 4)
            self.gemm(a, b, c, max_active_clusters, stream)

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
    mB = _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN)
    mC = _make_hopper_dense_tensor(c_view, c_cutlass_dtype)
    cluster_shape_mn = _hopper_dense_persistent_cluster_shape_mn(tile_shape_mn)
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    compiled = cute.compile(
        HopperWgmmaKBlockBatchedPersistentGemmKernel(),
        mA,
        mB,
        mC,
        max_active_clusters,
        stream,
    )
    _HOPPER_DENSE_PERSISTENT_KBLOCK_BATCHED_GEMM_COMPILED[key] = compiled
    return compiled


def _get_hopper_blockwise_scaled_persistent_gemm(
    a_view: torch.Tensor,
    b_view: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    c_view: torch.Tensor,
    key: tuple,
    tile_shape_mn: tuple[int, int],
    scale_k_blocks: int,
    wgrad: bool = False,
):
    cached = _HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED.get(key)
    if cached is not None:
        return cached

    module = _load_hopper_dense_gemm_persistent_module()
    if module is None:
        return None

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    pipeline = module.pipeline
    pipeline_init_arrive = module.pipeline_init_arrive
    pipeline_init_wait = module.pipeline_init_wait
    sm90_utils = module.sm90_utils
    utils = module.utils

    class HopperWgmmaBlockwiseScaledPersistentGemmKernel(
        module.HopperWgmmaGemmPersistentKernel
    ):
        def __init__(
            self,
            acc_dtype,
            tile_shape_mn,
            cluster_shape_mn,
            swizzle_size,
            raster_along_m,
            m_per_group,
            scale_k_blocks,
            wgrad,
        ):
            super().__init__(
                acc_dtype,
                tile_shape_mn,
                cluster_shape_mn,
                swizzle_size,
                raster_along_m,
            )
            self.m_per_group = m_per_group
            self.scale_k_blocks = scale_k_blocks
            self.wgrad = wgrad
            if (wgrad or scale_k_blocks == 56) and tile_shape_mn == (128, 128):
                self.atom_layout_mnk = (2, 1, 1)
                self.num_mma_warp_groups = 2
                self.threads_per_cta = 384
                self.num_mma_threads = 256
                self.epilog_sync_barrier = pipeline.NamedBarrier(
                    barrier_id=1,
                    num_threads=self.num_mma_threads,
                )

        def _setup_attributes(self):
            if self.tile_shape_mnk[1] != 192:
                return super()._setup_attributes()

            self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_layout.sm90_mma_major_mode(),
                self.b_layout.sm90_mma_major_mode(),
                self.acc_dtype,
                self.atom_layout_mnk,
                tiler_mn=(64, self.tile_shape_mnk[1]),
            )
            mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
            self.tile_shape_mnk = (
                self.tile_shape_mnk[0],
                self.tile_shape_mnk[1],
                mma_inst_shape_k * 4,
            )
            self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
            self.num_mcast_ctas_a = self.cluster_shape_mn[1]
            self.num_mcast_ctas_b = self.cluster_shape_mn[0]
            self.is_a_mcast = self.num_mcast_ctas_a > 1
            self.is_b_mcast = self.num_mcast_ctas_b > 1
            self.epi_tile = (128, 64)
            self.ab_stage = 4
            self.epi_stage = 2
            (
                self.a_smem_layout_staged,
                self.b_smem_layout_staged,
                self.epi_smem_layout_staged,
            ) = self._make_smem_layouts(
                self.tile_shape_mnk,
                self.epi_tile,
                self.a_dtype,
                self.a_layout,
                self.b_dtype,
                self.b_layout,
                self.ab_stage,
                self.c_dtype,
                self.c_layout,
                self.epi_stage,
            )

        @cute.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            a_scale: cute.Tensor,
            b_scale: cute.Tensor,
            c: cute.Tensor,
            max_active_clusters: cutlass.Constexpr,
            stream: cuda.CUstream,
        ):
            self.a_dtype = a.element_type
            self.b_dtype = b.element_type
            self.c_dtype = c.element_type
            self.a_layout = utils.LayoutEnum.from_tensor(a)
            self.b_layout = utils.LayoutEnum.from_tensor(b)
            self.c_layout = utils.LayoutEnum.from_tensor(c)

            self._setup_attributes()
            self.final_acc_dtype = (
                cutlass.Float32
                if self.wgrad
                or self.scale_k_blocks == 56
                or self.tile_shape_mnk[1] == 192
                else self.c_dtype
                if self.scale_k_blocks == 16
                else cutlass.Float16
                if self.scale_k_blocks == 56
                else self.acc_dtype
            )

            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                a,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[1],
            )
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                self.cluster_shape_mn[0],
            )
            tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
                c,
                self.epi_smem_layout_staged,
                self.epi_tile,
            )
            if cutlass.const_expr(self.wgrad):
                a_scale_smem_layout = cute.make_layout(
                    (1, self.tile_shape_mnk[0]),
                    stride=(self.tile_shape_mnk[0], 1),
                )
                b_scale_smem_layout = cute.make_layout(
                    (1, self.tile_shape_mnk[1]),
                    stride=(self.tile_shape_mnk[1], 1),
                )
                tma_atom_a_scale, tma_tensor_a_scale = (
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                        a_scale,
                        a_scale_smem_layout,
                        (1, self.tile_shape_mnk[0]),
                    )
                )
                tma_atom_b_scale, tma_tensor_b_scale = (
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                        b_scale,
                        b_scale_smem_layout,
                        (1, self.tile_shape_mnk[1]),
                    )
                )
            elif cutlass.const_expr(self.scale_k_blocks == 16):
                a_scale_smem_layout = cute.make_layout(
                    (
                        1 if self.tile_shape_mnk[1] == 192 else self.scale_k_blocks,
                        self.tile_shape_mnk[0],
                    ),
                    stride=(self.tile_shape_mnk[0], 1),
                )
                tma_atom_a_scale, tma_tensor_a_scale = (
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                        a_scale,
                        a_scale_smem_layout,
                        (
                            1 if self.tile_shape_mnk[1] == 192 else self.scale_k_blocks,
                            self.tile_shape_mnk[0],
                        ),
                    )
                )
                tma_atom_b_scale = tma_atom_a_scale
                tma_tensor_b_scale = tma_tensor_a_scale
            else:
                a_scale_smem_layout = cute.make_layout(
                    (1, self.tile_shape_mnk[0]),
                    stride=(self.tile_shape_mnk[0], 1),
                )
                tma_atom_a_scale, tma_tensor_a_scale = (
                    cute.nvgpu.cpasync.make_tiled_tma_atom(
                        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                        a_scale,
                        a_scale_smem_layout,
                        (1, self.tile_shape_mnk[0]),
                    )
                )
                tma_atom_b_scale = tma_atom_a_scale
                tma_tensor_b_scale = tma_tensor_a_scale
            tile_sched_params, grid = self._compute_grid(
                c,
                self.tile_shape_mnk,
                self.cluster_shape_mn,
                self.swizzle_size,
                self.raster_along_m,
                max_active_clusters,
            )
            separate_a_scale_elems = 2 * self.scale_k_blocks * self.tile_shape_mnk[0]
            separate_b_scale_elems = self.scale_k_blocks * (
                (self.tile_shape_mnk[1] + 127) // 128
            )
            if cutlass.const_expr(self.wgrad):
                separate_a_scale_elems = self.ab_stage * self.tile_shape_mnk[0]
                separate_b_scale_elems = self.ab_stage * self.tile_shape_mnk[1]
            elif cutlass.const_expr(self.tile_shape_mnk[1] == 192):
                separate_a_scale_elems = self.ab_stage * self.tile_shape_mnk[0]

            @cute.struct
            class SeparateScaleSharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype,
                        cute.cosize(self.epi_smem_layout_staged),
                    ],
                    self.buffer_align_bytes,
                ]
                sAScale: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float32,
                        separate_a_scale_elems,
                    ],
                    self.buffer_align_bytes,
                ]
                sBScale: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float32,
                        separate_b_scale_elems,
                    ],
                    self.buffer_align_bytes,
                ]

            @cute.struct
            class ProductScaleSharedStorage:
                mainloop_pipeline_array_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.ab_stage * 2
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[
                        self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[
                        self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                    ],
                    self.buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        self.c_dtype,
                        cute.cosize(self.epi_smem_layout_staged),
                    ],
                    self.buffer_align_bytes,
                ]
                sAScale: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float32,
                        self.ab_stage * self.tile_shape_mnk[0],
                    ],
                    self.buffer_align_bytes,
                ]
                sBScale: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Float32,
                        self.scale_k_blocks * ((self.tile_shape_mnk[1] + 127) // 128),
                    ],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = (
                SeparateScaleSharedStorage
                if self.wgrad or self.scale_k_blocks == 16
                else ProductScaleSharedStorage
            )

            if cutlass.const_expr(self.scale_k_blocks == 16):
                kernel = self.kernel(
                    tma_atom_a,
                    tma_tensor_a,
                    tma_atom_b,
                    tma_tensor_b,
                    tma_atom_c,
                    tma_tensor_c,
                    tma_atom_a_scale,
                    tma_tensor_a_scale,
                    tma_atom_b_scale,
                    tma_tensor_b_scale,
                    a_scale,
                    b_scale,
                    self.tiled_mma,
                    self.cta_layout_mnk,
                    self.a_smem_layout_staged,
                    self.b_smem_layout_staged,
                    self.epi_smem_layout_staged,
                    tile_sched_params,
                    M_PER_GROUP=self.m_per_group,
                    BLOCK_SIZE=128,
                    SCALE_K_BLOCKS=self.scale_k_blocks,
                    K_TILES_PER_SCALE_BLOCK=128 // self.tile_shape_mnk[2],
                )
            else:
                kernel = self.kernel(
                    tma_atom_a,
                    tma_tensor_a,
                    tma_atom_b,
                    tma_tensor_b,
                    tma_atom_c,
                    tma_tensor_c,
                    tma_atom_a_scale,
                    tma_tensor_a_scale,
                    tma_atom_b_scale,
                    tma_tensor_b_scale,
                    a_scale,
                    b_scale,
                    self.tiled_mma,
                    self.cta_layout_mnk,
                    self.a_smem_layout_staged,
                    self.b_smem_layout_staged,
                    self.epi_smem_layout_staged,
                    tile_sched_params,
                    M_PER_GROUP=self.m_per_group,
                    BLOCK_SIZE=128,
                    SCALE_K_BLOCKS=self.scale_k_blocks,
                    K_TILES_PER_SCALE_BLOCK=128 // self.tile_shape_mnk[2],
                )
            kernel.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                min_blocks_per_mp=1,
                stream=stream,
            )

        @staticmethod
        def layout_separate(thr, src, ref):
            lt = cute.make_layout(())
            ge = cute.make_layout(())

            for k, v in enumerate(ref):
                if cutlass.const_expr(v < thr):
                    lt = cute.append(lt, src[k])
                else:
                    ge = cute.append(ge, src[k])
            return lt, ge

        @cute.jit
        def layout_acc_mn(self, tiled_mma, acc):
            separated = self.layout_separate(
                tiled_mma.shape_mnk[0],
                acc[0],
                tiled_mma.tv_layout_C.stride[1],
            )

            v_m = separated[0]
            v_n = separated[1]
            if cutlass.const_expr(cute.rank(v_m) == 1):
                v_m1 = cute.append(v_m, acc[1])
            else:
                v_m1 = cute.append(cute.append(cute.make_layout(()), v_m), acc[1])

            if cutlass.const_expr(cute.rank(v_n) == 1):
                v_n1 = cute.append(v_n, acc[2])
            else:
                v_n1 = cute.append(cute.append(cute.make_layout(()), v_n), acc[2])

            if cutlass.const_expr(cute.rank(v_m1) == 1):
                return cute.append(v_m1, v_n1)
            return cute.append(cute.append(cute.make_layout(()), v_m1), v_n1)

        def _compute_stages(
            self,
            tile_shape_mnk,
            a_dtype,
            b_dtype,
            epi_tile,
            c_dtype,
            smem_capacity,
            occupancy,
        ):
            if self.wgrad:
                base_stages, _ = module.HopperWgmmaGemmPersistentKernel._compute_stages(
                    tile_shape_mnk,
                    a_dtype,
                    b_dtype,
                    epi_tile,
                    c_dtype,
                    smem_capacity,
                    occupancy,
                )
                scale_smem_bytes = (
                    base_stages
                    * (tile_shape_mnk[0] + tile_shape_mnk[1])
                    * cutlass.Float32.width
                    // 8
                )
                return module.HopperWgmmaGemmPersistentKernel._compute_stages(
                    tile_shape_mnk,
                    a_dtype,
                    b_dtype,
                    epi_tile,
                    c_dtype,
                    smem_capacity - scale_smem_bytes,
                    occupancy,
                )
            scale_smem_elems = (
                self.scale_k_blocks * (2 * tile_shape_mnk[0] + tile_shape_mnk[1] // 128)
                if self.scale_k_blocks == 16
                else self.scale_k_blocks
                * tile_shape_mnk[0]
                * ((tile_shape_mnk[1] + 127) // 128)
            )
            scale_smem_width = (
                cutlass.Float32.width
                if self.wgrad or self.scale_k_blocks == 16
                else cutlass.Float16.width
            )
            scale_smem_bytes = scale_smem_elems * scale_smem_width // 8
            return module.HopperWgmmaGemmPersistentKernel._compute_stages(
                tile_shape_mnk,
                a_dtype,
                b_dtype,
                epi_tile,
                c_dtype,
                smem_capacity - scale_smem_bytes,
                occupancy,
            )

        @cute.kernel
        def kernel(
            self,
            tma_atom_a: cute.CopyAtom,
            mA_mkl: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB_nkl: cute.Tensor,
            tma_atom_c: cute.CopyAtom,
            mC_mnl: cute.Tensor,
            tma_atom_a_scale: cute.CopyAtom,
            mAScale_km: cute.Tensor,
            tma_atom_b_scale: cute.CopyAtom,
            mBScale_knl: cute.Tensor,
            a_scale: cute.Tensor,
            b_scale: cute.Tensor,
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_smem_layout_staged: cute.ComposedLayout,
            tile_sched_params: utils.PersistentTileSchedulerParams,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            SCALE_K_BLOCKS: cutlass.Constexpr[int],
            K_TILES_PER_SCALE_BLOCK: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
            if warp_idx == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a_scale)
                if cutlass.const_expr(self.wgrad):
                    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_scale)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

            a_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=1
            )
            b_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=0
            )
            a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
            b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

            a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
            b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
            tma_copy_bytes = cute.size_in_bytes(
                self.a_dtype, a_smem_layout
            ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            if cutlass.const_expr(self.wgrad):
                tma_copy_bytes += (
                    (self.tile_shape_mnk[0] + self.tile_shape_mnk[1])
                    * cutlass.Float32.width
                    // 8
                )

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            consumer_arrive_cnt = (
                mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
            )
            mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=mainloop_pipeline_array_ptr,
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                tx_count=tma_copy_bytes,
                cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
                defer_sync=True,
            )

            pipeline_init_arrive(
                cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
            )

            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            sC = storage.sC.get_tensor(
                epi_smem_layout_staged.outer,
                swizzle=epi_smem_layout_staged.inner,
            )
            scale_blocks_n = (
                self.tile_shape_mnk[1]
                if self.wgrad
                else (self.tile_shape_mnk[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
            )
            if cutlass.const_expr(
                SCALE_K_BLOCKS == 16 and self.tile_shape_mnk[1] != 192
            ):
                a_scale_k_stride = self.tile_shape_mnk[0]
                b_scale_k_stride = scale_blocks_n
                a_scale_cache_elems = SCALE_K_BLOCKS * a_scale_k_stride
                b_scale_cache_elems = SCALE_K_BLOCKS * b_scale_k_stride
                sAScale = storage.sAScale.get_tensor(
                    cute.make_layout(
                        (SCALE_K_BLOCKS, self.tile_shape_mnk[0]),
                        stride=(a_scale_k_stride, 1),
                    )
                )
                sBScale = storage.sBScale.get_tensor(
                    cute.make_layout(
                        (SCALE_K_BLOCKS, scale_blocks_n),
                        stride=(b_scale_k_stride, 1),
                    )
                )
            else:
                b_scale_k_stride = scale_blocks_n
                b_scale_cache_elems = SCALE_K_BLOCKS * b_scale_k_stride
                a_scale_tma_bytes = self.tile_shape_mnk[0] * cutlass.Float32.width // 8
                sAScale = storage.sAScale.get_tensor(
                    cute.make_layout(
                        (1, self.tile_shape_mnk[0], self.ab_stage),
                        stride=(
                            self.tile_shape_mnk[0],
                            1,
                            self.tile_shape_mnk[0],
                        ),
                    )
                )
                sBScale = storage.sBScale.get_tensor(
                    cute.make_layout(
                        (SCALE_K_BLOCKS, scale_blocks_n),
                        stride=(b_scale_k_stride, 1),
                    )
                )

            gA_mkl = cute.local_tile(
                mA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            gB_nkl = cute.local_tile(
                mB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            gC_mnl = cute.local_tile(
                mC_mnl,
                cute.slice_(self.tile_shape_mnk, (None, None, 0)),
                (None, None, None),
            )
            if cutlass.const_expr(self.wgrad):
                gAScale_mkl = cute.local_tile(
                    mAScale_km,
                    (self.tile_shape_mnk[0], 1),
                    (None, None, None),
                )
                gBScale_nkl = cute.local_tile(
                    mBScale_knl,
                    (self.tile_shape_mnk[1], 1),
                    (None, None, None),
                )
            if cutlass.const_expr(
                SCALE_K_BLOCKS == 56 or self.tile_shape_mnk[1] == 192
            ):
                gAScale_km = cute.local_tile(
                    mAScale_km,
                    (1, self.tile_shape_mnk[0]),
                    (None, None),
                )
            a_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (0, None, 0)).shape
            )
            a_cta_crd = cluster_coord_mnk[1]
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

            b_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
            )
            b_cta_crd = cluster_coord_mnk[0]
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )
            if cutlass.const_expr(self.wgrad):
                tSsAScaleWgrad, tSgAScaleWgrad = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_a_scale,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sAScale, 0, 2),
                    cute.group_modes(gAScale_mkl, 0, 2),
                )
                tSsBScaleWgrad, tSgBScaleWgrad = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_b_scale,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sBScale, 0, 2),
                    cute.group_modes(gBScale_nkl, 0, 2),
                )
            if cutlass.const_expr(
                SCALE_K_BLOCKS == 56 or self.tile_shape_mnk[1] == 192
            ):
                tSsAScale, tSgAScale = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_a_scale,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sAScale, 0, 2),
                    cute.group_modes(gAScale_km, 0, 2),
                )

            warp_group_idx = cute.arch.make_warp_uniform(
                tidx // self.num_threads_per_warp_group
            )
            mma_warp_group_thread_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(
                mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
            )

            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)

            tCgC = thr_mma.partition_C(gC_mnl)
            acc_shape = tCgC.shape[:3]
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            k_tile_cnt = cute.size(gA_mkl, mode=[3])
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
            if is_dma_warp_group:
                cute.arch.setmaxregister_decrease(self.load_register_requirement)

            if warp_idx == self.load_warp_id:
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
                work_tile = tile_sched.initial_work_tile_info()
                mainloop_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )

                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                    if cutlass.const_expr(
                        SCALE_K_BLOCKS == 56 or self.tile_shape_mnk[1] == 192
                    ):
                        expert = cutlass.Int64(tile_coord_mnl[2])
                        a_scale_tile_m = (
                            expert * (M_PER_GROUP // self.tile_shape_mnk[0])
                            + tile_coord_mnl[0]
                        )
                    mainloop_producer_state.reset_count()

                    for k_tile in range(k_tile_cnt):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                        tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                        mainloop_barrier = mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        )

                        if cutlass.const_expr(
                            SCALE_K_BLOCKS == 56 or self.tile_shape_mnk[1] == 192
                        ):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_expect_tx(
                                    mainloop_barrier,
                                    a_scale_tma_bytes,
                                )
                            cute.copy(
                                tma_atom_a_scale,
                                tSgAScale[
                                    (
                                        None,
                                        mainloop_producer_state.count,
                                        a_scale_tile_m,
                                    )
                                ],
                                tSsAScale[(None, mainloop_producer_state.index)],
                                tma_bar_ptr=mainloop_barrier,
                            )

                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_barrier,
                            mcast_mask=b_mcast_mask,
                        )

                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                mainloop_pipeline.producer_tail(mainloop_producer_state)

            if not is_dma_warp_group:
                cute.arch.setmaxregister_increase(self.mma_register_requirement)
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                )
                work_tile = tile_sched.initial_work_tile_info()

                mainloop_consumer_read_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                mainloop_consumer_release_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                num_k_blocks = cute.size(tCrA, mode=[2])
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    self.c_dtype,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )
                mma_tidx = (
                    tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
                )
                thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                thr_mapping = cute.make_identity_tensor(self.epi_tile)
                tRS_cS = thr_copy_r2s.partition_S(thr_mapping)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_threads,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                    tCgC_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )
                    epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                    epi_tile_shape = tCgC_for_tma_partition.shape[1]
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(epi_tile_shape[1], 1)
                    )
                    tRS_rFinal = cute.make_rmem_tensor(
                        (epi_tile_num, size_tRS_rD), self.final_acc_dtype
                    )

                    expert = cutlass.Int64(tile_coord_mnl[2])
                    a_scale_row_base = (
                        expert * M_PER_GROUP
                        + tile_coord_mnl[0] * self.tile_shape_mnk[0]
                    )
                    if cutlass.const_expr(self.tile_shape_mnk[1] == 192):
                        tile_n_start = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                        n_block_base = tile_n_start // BLOCK_SIZE
                    else:
                        n_block_base = tile_coord_mnl[1] * scale_blocks_n
                        tile_n_scale_offset = cutlass.Int32(0)
                    if cutlass.const_expr(
                        SCALE_K_BLOCKS == 16 and self.tile_shape_mnk[1] != 192
                    ):
                        for scale_load_idx in cutlass.range(
                            mma_tidx,
                            a_scale_cache_elems,
                            self.num_mma_threads,
                            unroll=1,
                        ):
                            scale_k = scale_load_idx // a_scale_k_stride
                            scale_cache_rem = (
                                scale_load_idx - scale_k * a_scale_k_stride
                            )
                            scale_m = scale_cache_rem
                            m = cutlass.Int64(a_scale_row_base + scale_m)
                            sAScale[scale_k, scale_m] = cutlass.Float32(
                                a_scale[scale_k, m]
                            )
                        for scale_load_idx in cutlass.range(
                            mma_tidx,
                            b_scale_cache_elems,
                            self.num_mma_threads,
                            unroll=1,
                        ):
                            scale_k = scale_load_idx // b_scale_k_stride
                            scale_cache_rem = (
                                scale_load_idx - scale_k * b_scale_k_stride
                            )
                            scale_n = scale_cache_rem
                            n_block = cutlass.Int64(n_block_base + scale_n)
                            sBScale[scale_k, scale_n] = cutlass.Float32(
                                b_scale[expert, scale_k, n_block]
                            )
                    else:
                        for scale_load_idx in cutlass.range(
                            mma_tidx,
                            b_scale_cache_elems,
                            self.num_mma_threads,
                            unroll=1,
                        ):
                            scale_k = scale_load_idx // b_scale_k_stride
                            scale_cache_rem = (
                                scale_load_idx - scale_k * b_scale_k_stride
                            )
                            scale_n = scale_cache_rem
                            n_block = cutlass.Int64(
                                min(
                                    n_block_base + scale_n,
                                    b_scale.shape[2] - 1,
                                )
                            )
                            sBScale[scale_k, scale_n] = cutlass.Float32(
                                b_scale[expert, scale_k, n_block]
                            )
                    self.epilog_sync_barrier.arrive_and_wait()

                    tRS_rFinal.fill(0.0)
                    mainloop_consumer_read_state.reset_count()
                    mainloop_consumer_release_state.reset_count()

                    for k_tile in cutlass.range(
                        0,
                        k_tile_cnt // K_TILES_PER_SCALE_BLOCK,
                        1,
                        unroll=8,
                    ):
                        accumulators.fill(0.0)
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                        cute.nvgpu.warpgroup.fence()
                        scale_k = (
                            mainloop_consumer_read_state.count
                            // K_TILES_PER_SCALE_BLOCK
                        )

                        for _ in cutlass.range_constexpr(K_TILES_PER_SCALE_BLOCK):
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_read_state
                            )
                            if cutlass.const_expr(
                                SCALE_K_BLOCKS == 56 or self.tile_shape_mnk[1] == 192
                            ):
                                fwd_scale_a0 = sAScale[
                                    0,
                                    tRS_cS[0][0],
                                    mainloop_consumer_read_state.index,
                                ]
                                fwd_scale_a1 = sAScale[
                                    0,
                                    tRS_cS[2][0],
                                    mainloop_consumer_read_state.index,
                                ]
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_block_coord = (
                                    None,
                                    None,
                                    k_block_idx,
                                    mainloop_consumer_read_state.index,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[k_block_coord],
                                    tCrB[k_block_coord],
                                    accumulators,
                                )

                            cute.nvgpu.warpgroup.commit_group()
                            cute.nvgpu.warpgroup.wait_group(0)
                            mainloop_pipeline.consumer_release(
                                mainloop_consumer_release_state
                            )
                            mainloop_consumer_release_state.advance()
                            mainloop_consumer_read_state.advance()
                        if cutlass.const_expr(
                            SCALE_K_BLOCKS == 16 and self.tile_shape_mnk[1] == 192
                        ):
                            scale_b0 = sBScale[scale_k, 0]
                            scale_b1 = sBScale[scale_k, 1]
                            scale_0_0 = fwd_scale_a0 * scale_b0
                            scale_1_0 = fwd_scale_a1 * scale_b0
                            scale_0_1 = fwd_scale_a0 * scale_b1
                            scale_1_1 = fwd_scale_a1 * scale_b1
                            if tile_n_scale_offset == 0:
                                for epi_v in cutlass.range_constexpr(96):
                                    acc_group = epi_v // 4
                                    if cutlass.const_expr(acc_group < 16):
                                        scale_0 = scale_0_0
                                        scale_1 = scale_1_0
                                    else:
                                        scale_0 = scale_0_1
                                        scale_1 = scale_1_1
                                    scale = (
                                        scale_0
                                        if cutlass.const_expr((epi_v // 2) % 2 == 0)
                                        else scale_1
                                    )
                                    tRS_rFinal[0, epi_v] = (
                                        tRS_rFinal[0, epi_v] + tRS_rAcc[epi_v] * scale
                                    )
                            else:
                                for epi_v in cutlass.range_constexpr(96):
                                    acc_group = epi_v // 4
                                    if cutlass.const_expr(acc_group < 8):
                                        scale_0 = scale_0_0
                                        scale_1 = scale_1_0
                                    else:
                                        scale_0 = scale_0_1
                                        scale_1 = scale_1_1
                                    scale = (
                                        scale_0
                                        if cutlass.const_expr((epi_v // 2) % 2 == 0)
                                        else scale_1
                                    )
                                    tRS_rFinal[0, epi_v] = (
                                        tRS_rFinal[0, epi_v] + tRS_rAcc[epi_v] * scale
                                    )
                        else:
                            for subtile_idx in cutlass.range_constexpr(epi_tile_num):
                                epi_coord = epi_tile_layout.get_hier_coord(subtile_idx)
                                if cutlass.const_expr(SCALE_K_BLOCKS == 56):
                                    fwd_scale_n = (
                                        tile_n_scale_offset
                                        + epi_coord[1] * self.epi_tile[1]
                                    ) // BLOCK_SIZE
                                    fwd_scale_b = sBScale[scale_k, fwd_scale_n]
                                    fwd_scale_0 = fwd_scale_a0 * fwd_scale_b
                                    fwd_scale_1 = fwd_scale_a1 * fwd_scale_b
                                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                                    acc_i = subtile_idx * size_tRS_rD + epi_v
                                    coord = tRS_cS[epi_v]
                                    if cutlass.const_expr(SCALE_K_BLOCKS == 16):
                                        scale_m = (
                                            epi_coord[0] * self.epi_tile[0] + coord[0]
                                        )
                                        scale_n = (
                                            epi_coord[1] * self.epi_tile[1]
                                            + cutlass.Int64(coord[1])
                                        ) // BLOCK_SIZE
                                        scale = (
                                            sAScale[scale_k, scale_m]
                                            * sBScale[scale_k, scale_n]
                                        )
                                    else:
                                        if cutlass.const_expr((epi_v // 2) % 2 == 0):
                                            scale = fwd_scale_0
                                        else:
                                            scale = fwd_scale_1
                                    partial = tRS_rAcc[acc_i]
                                    if cutlass.const_expr(SCALE_K_BLOCKS == 16):
                                        final = tRS_rFinal[
                                            subtile_idx, epi_v
                                        ] + partial.to(self.final_acc_dtype) * scale.to(
                                            self.final_acc_dtype
                                        )
                                    else:
                                        final = (
                                            tRS_rFinal[subtile_idx, epi_v]
                                            + partial * scale
                                        )
                                    tRS_rFinal[subtile_idx, epi_v] = final.to(
                                        self.final_acc_dtype
                                    )

                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        tCgC_for_tma_partition,
                    )
                    num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                    for epi_idx in cutlass.range_constexpr(epi_tile_num):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            tRS_rD[epi_v] = tRS_rFinal[epi_idx, epi_v]

                        acc_vec = tRS_rD.load()
                        tRS_rD_out.store(acc_vec.to(self.c_dtype))

                        epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                        if warp_idx == self.epi_store_warp_id:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, gmem_coord)],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()

                        self.epilog_sync_barrier.arrive_and_wait()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                tma_store_pipeline.producer_tail()

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
    mB = _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN)
    mAS = _make_cutedsl_tensor(a_s, cutlass.Float32, leading_dim=1)
    mBS = _make_cutedsl_tensor(
        b_s,
        cutlass.Float32,
        leading_dim=1,
    )
    mC = _make_hopper_dense_tensor(c_view, cutlass.BFloat16)
    cluster_shape_mn = _hopper_dense_persistent_cluster_shape_mn(tile_shape_mn)
    if wgrad:
        cluster_shape_mn = (1, 1)
    elif tile_shape_mn == _HOPPER_DENSE_FWD_TILE_SHAPE_MN:
        cluster_shape_mn = (2, 1)
    elif tile_shape_mn == (128, 256):
        cluster_shape_mn = (1, 2) if scale_k_blocks == 16 else (1, 1)
    swizzle_size = 1

    # Distinct class identities keep fwd-only codegen changes from perturbing dgrad.
    class HopperWgmmaBlockwiseScaledPersistentFwdGemmKernel(
        HopperWgmmaBlockwiseScaledPersistentGemmKernel
    ):
        pass

    class HopperWgmmaBlockwiseScaledPersistentDgradGemmKernel(
        HopperWgmmaBlockwiseScaledPersistentGemmKernel
    ):
        @cute.kernel
        def kernel(
            self,
            tma_atom_a: cute.CopyAtom,
            mA_mkl: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB_nkl: cute.Tensor,
            tma_atom_c: cute.CopyAtom,
            mC_mnl: cute.Tensor,
            tma_atom_a_scale: cute.CopyAtom,
            mAScale_km: cute.Tensor,
            tma_atom_b_scale: cute.CopyAtom,
            mBScale_knl: cute.Tensor,
            a_scale: cute.Tensor,
            b_scale: cute.Tensor,
            tiled_mma: cute.TiledMma,
            cta_layout_mnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            epi_smem_layout_staged: cute.ComposedLayout,
            tile_sched_params: utils.PersistentTileSchedulerParams,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            SCALE_K_BLOCKS: cutlass.Constexpr[int],
            K_TILES_PER_SCALE_BLOCK: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)

            if warp_idx == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a_scale)
                if cutlass.const_expr(self.wgrad):
                    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_scale)

            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

            a_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=1
            )
            b_mcast_mask = cute.make_layout_image_mask(
                cta_layout_mnk, cluster_coord_mnk, mode=0
            )
            a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
            b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

            a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
            b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
            tma_copy_bytes = cute.size_in_bytes(
                self.a_dtype, a_smem_layout
            ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            if cutlass.const_expr(self.wgrad):
                tma_copy_bytes += (
                    (self.tile_shape_mnk[0] + self.tile_shape_mnk[1])
                    * cutlass.Float32.width
                    // 8
                )

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

            mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            consumer_arrive_cnt = (
                mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
            )
            mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_arrive_cnt
            )
            mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=mainloop_pipeline_array_ptr,
                num_stages=self.ab_stage,
                producer_group=mainloop_pipeline_producer_group,
                consumer_group=mainloop_pipeline_consumer_group,
                tx_count=tma_copy_bytes,
                cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
                defer_sync=True,
            )

            pipeline_init_arrive(
                cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
            )

            sA = storage.sA.get_tensor(
                a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
            )
            sB = storage.sB.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            sC = storage.sC.get_tensor(
                epi_smem_layout_staged.outer,
                swizzle=epi_smem_layout_staged.inner,
            )
            scale_blocks_n = (
                self.tile_shape_mnk[1]
                if self.wgrad
                else (self.tile_shape_mnk[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
            )
            a_scale_k_stride = self.tile_shape_mnk[0]
            b_scale_k_stride = scale_blocks_n
            a_scale_cache_elems = (
                self.tile_shape_mnk[0]
                if self.wgrad
                else SCALE_K_BLOCKS * a_scale_k_stride
            )
            b_scale_cache_elems = (
                self.tile_shape_mnk[1]
                if self.wgrad
                else SCALE_K_BLOCKS * b_scale_k_stride
            )
            a_scale_tma_bytes = (
                0 if self.wgrad else a_scale_cache_elems * cutlass.Float32.width // 8
            )
            # The DMA producer can advance into the next persistent work tile
            # before MMA threads finish the current tile's scaled epilogue.
            if cutlass.const_expr(self.wgrad):
                sAScale = storage.sAScale.get_tensor(
                    cute.make_layout(
                        (1, self.tile_shape_mnk[0], self.ab_stage),
                        stride=(self.tile_shape_mnk[0], 1, self.tile_shape_mnk[0]),
                    )
                )
                sBScale = storage.sBScale.get_tensor(
                    cute.make_layout(
                        (1, self.tile_shape_mnk[1], self.ab_stage),
                        stride=(self.tile_shape_mnk[1], 1, self.tile_shape_mnk[1]),
                    )
                )
            else:
                sAScale = storage.sAScale.get_tensor(
                    cute.make_layout(
                        (2, SCALE_K_BLOCKS, self.tile_shape_mnk[0]),
                        stride=(a_scale_cache_elems, a_scale_k_stride, 1),
                    )
                )
                sBScale = storage.sBScale.get_tensor(
                    cute.make_layout(
                        (SCALE_K_BLOCKS, scale_blocks_n),
                        stride=(b_scale_k_stride, 1),
                    )
                )

            gA_mkl = cute.local_tile(
                mA_mkl,
                cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                (None, None, None),
            )
            gB_nkl = cute.local_tile(
                mB_nkl,
                cute.slice_(self.tile_shape_mnk, (0, None, None)),
                (None, None, None),
            )
            gC_mnl = cute.local_tile(
                mC_mnl,
                cute.slice_(self.tile_shape_mnk, (None, None, 0)),
                (None, None, None),
            )
            if cutlass.const_expr(self.wgrad):
                gAScale_mkl = cute.local_tile(
                    mAScale_km,
                    (1, self.tile_shape_mnk[0]),
                    (None, None),
                )
                gBScale_nkl = cute.local_tile(
                    mBScale_knl,
                    (1, self.tile_shape_mnk[1]),
                    (None, None),
                )
            a_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (0, None, 0)).shape
            )
            a_cta_crd = cluster_coord_mnk[1]
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_atom_a,
                a_cta_crd,
                a_cta_layout,
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA_mkl, 0, 2),
            )

            b_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
            )
            b_cta_crd = cluster_coord_mnk[0]
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )
            if cutlass.const_expr(self.wgrad):
                tSsAScaleWgrad, tSgAScaleWgrad = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_a_scale,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sAScale, 0, 2),
                    cute.group_modes(gAScale_mkl, 0, 2),
                )
                tSsBScaleWgrad, tSgBScaleWgrad = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_b_scale,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sBScale, 0, 2),
                    cute.group_modes(gBScale_nkl, 0, 2),
                )

            warp_group_idx = cute.arch.make_warp_uniform(
                tidx // self.num_threads_per_warp_group
            )
            mma_warp_group_thread_layout = cute.make_layout(
                self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(
                mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
            )

            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)

            tCgC = thr_mma.partition_C(gC_mnl)
            acc_shape = tCgC.shape[:3]
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            k_tile_cnt = cute.size(gA_mkl, mode=[3])
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

            is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
            if is_dma_warp_group:
                cute.arch.setmaxregister_decrease(self.load_register_requirement)

            if warp_idx == self.load_warp_id:
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                mainloop_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )

                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                    if cutlass.const_expr(not self.wgrad):
                        scale_buffer = tile_sched.num_tiles_executed % 2
                        sAScale_write = sAScale[(scale_buffer, None, None)]
                        expert = cutlass.Int64(tile_coord_mnl[2])
                        a_scale_tile_m = (
                            expert * (M_PER_GROUP // self.tile_shape_mnk[0])
                            + tile_coord_mnl[0]
                        )
                        gAScale_km = cute.local_tile(
                            mAScale_km,
                            (SCALE_K_BLOCKS, self.tile_shape_mnk[0]),
                            (0, a_scale_tile_m),
                        )
                        tSsAScale, tSgAScale = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_a_scale,
                            0,
                            cute.make_layout((1,)),
                            sAScale_write,
                            gAScale_km,
                        )
                    mainloop_producer_state.reset_count()

                    for k_tile in range(k_tile_cnt):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                        tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                        mainloop_barrier = mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        )

                        if cutlass.const_expr(self.wgrad):
                            scale_col = cutlass.Int64(
                                tile_coord_mnl[2]
                            ) * SCALE_K_BLOCKS + cutlass.Int64(
                                mainloop_producer_state.count
                            )
                            cute.copy(
                                tma_atom_a_scale,
                                tSgAScaleWgrad[
                                    (
                                        None,
                                        scale_col,
                                        tile_coord_mnl[0],
                                    )
                                ],
                                tSsAScaleWgrad[(None, mainloop_producer_state.index)],
                                tma_bar_ptr=mainloop_barrier,
                            )
                            cute.copy(
                                tma_atom_b_scale,
                                tSgBScaleWgrad[
                                    (
                                        None,
                                        scale_col,
                                        tile_coord_mnl[1],
                                    )
                                ],
                                tSsBScaleWgrad[(None, mainloop_producer_state.index)],
                                tma_bar_ptr=mainloop_barrier,
                            )
                        if cutlass.const_expr(not self.wgrad):
                            if mainloop_producer_state.count == 0:
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_expect_tx(
                                        mainloop_barrier,
                                        a_scale_tma_bytes,
                                    )
                                cute.copy(
                                    tma_atom_a_scale,
                                    tSgAScale[((None, None), 0)],
                                    tSsAScale[((None, None), 0)],
                                    tma_bar_ptr=mainloop_barrier,
                                )

                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_barrier,
                            mcast_mask=b_mcast_mask,
                        )

                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                mainloop_pipeline.producer_tail(mainloop_producer_state)

            if not is_dma_warp_group:
                cute.arch.setmaxregister_increase(self.mma_register_requirement)
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                mainloop_consumer_read_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                mainloop_consumer_release_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                num_k_blocks = cute.size(tCrA, mode=[2])
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    self.c_dtype,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )
                mma_tidx = (
                    tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
                )
                thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                thr_mapping = cute.make_identity_tensor(self.epi_tile)
                tRS_cS = thr_copy_r2s.partition_S(thr_mapping)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                size_tRS_rD = cute.size(tRS_rD)
                if cutlass.const_expr(self.wgrad):
                    scale_thread = mma_tidx % self.num_threads_per_warp_group
                    scale_warp = scale_thread // 32
                    scale_lane = scale_thread % 32
                    scale_row_0 = (
                        (mma_tidx // self.num_threads_per_warp_group) * 64
                        + scale_warp * 16
                        + scale_lane // 4
                    )
                    scale_row_1 = scale_row_0 + 8
                    scale_col_pair = (scale_lane % 4) * 2
                    scale_b_regs = cute.make_rmem_tensor(
                        (cute.size(tRS_rAcc) // 2,), cutlass.Float32
                    )

                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_threads,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                    tCgC_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )
                    epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                    epi_tile_shape = tCgC_for_tma_partition.shape[1]
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(epi_tile_shape[1], 1)
                    )
                    tRS_rFinal = cute.make_rmem_tensor(
                        (epi_tile_num, size_tRS_rD), self.final_acc_dtype
                    )

                    expert = cutlass.Int64(tile_coord_mnl[2])
                    scale_buffer = tile_sched.num_tiles_executed % 2
                    if cutlass.const_expr(self.tile_shape_mnk[1] == 192):
                        tile_n_start = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                        n_block_base = tile_n_start // BLOCK_SIZE
                    else:
                        n_block_base = tile_coord_mnl[1] * scale_blocks_n
                    if cutlass.const_expr(not self.wgrad):
                        for scale_load_idx in cutlass.range(
                            mma_tidx,
                            b_scale_cache_elems,
                            self.num_mma_threads,
                            unroll=1,
                        ):
                            scale_k = scale_load_idx // b_scale_k_stride
                            scale_cache_rem = (
                                scale_load_idx - scale_k * b_scale_k_stride
                            )
                            scale_n = scale_cache_rem
                            n_block = cutlass.Int64(
                                min(
                                    n_block_base + scale_n,
                                    b_scale.shape[2] - 1,
                                )
                            )
                            sBScale[scale_k, scale_n] = cutlass.Float32(
                                b_scale[expert, scale_k, n_block]
                            )
                        self.epilog_sync_barrier.arrive_and_wait()

                    mainloop_consumer_read_state.reset_count()
                    mainloop_consumer_release_state.reset_count()

                    # Initialize from scale block zero to avoid a zero-fill and
                    # first read/add of the final accumulator.
                    accumulators.fill(0.0)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.fence()
                    scale_k = (
                        mainloop_consumer_read_state.count // K_TILES_PER_SCALE_BLOCK
                    )

                    for _ in cutlass.range_constexpr(K_TILES_PER_SCALE_BLOCK):
                        mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                        if cutlass.const_expr(self.wgrad):
                            scale_stage = mainloop_consumer_read_state.index
                            scale_a_0 = sAScale[0, scale_row_0, scale_stage]
                            scale_a_1 = sAScale[0, scale_row_1, scale_stage]
                            for scale_i in cutlass.range_constexpr(
                                cute.size(tRS_rAcc) // 4
                            ):
                                scale_n = scale_i * 8 + scale_col_pair
                                scale_b_regs[scale_i * 2] = sBScale[
                                    0, scale_n, scale_stage
                                ]
                                scale_b_regs[scale_i * 2 + 1] = sBScale[
                                    0, scale_n + 1, scale_stage
                                ]
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (
                                None,
                                None,
                                k_block_idx,
                                mainloop_consumer_read_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[k_block_coord],
                                tCrB[k_block_coord],
                                accumulators,
                            )

                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(0)
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_release_state
                        )
                        mainloop_consumer_release_state.advance()
                        mainloop_consumer_read_state.advance()
                    if cutlass.const_expr(self.wgrad):
                        for acc_i in cutlass.range_constexpr(cute.size(tRS_rAcc)):
                            scale_i = acc_i // 4
                            scale_a = (
                                scale_a_0
                                if cutlass.const_expr(acc_i % 4 < 2)
                                else scale_a_1
                            )
                            scale_b = scale_b_regs[scale_i * 2 + acc_i % 2]
                            epi_idx = acc_i // size_tRS_rD
                            epi_v = acc_i - epi_idx * size_tRS_rD
                            tRS_rFinal[epi_idx, epi_v] = (
                                tRS_rAcc[acc_i] * scale_a * scale_b
                            )
                    else:
                        scale_a0 = sAScale[scale_buffer, scale_k, tRS_cS[0][0]]
                        scale_a1 = sAScale[scale_buffer, scale_k, tRS_cS[2][0]]
                        scale_0_0 = (scale_a0 * sBScale[scale_k, 0]).to(
                            self.final_acc_dtype
                        )
                        scale_1_0 = (scale_a1 * sBScale[scale_k, 0]).to(
                            self.final_acc_dtype
                        )
                        scale_0_1 = (scale_a0 * sBScale[scale_k, 1]).to(
                            self.final_acc_dtype
                        )
                        scale_1_1 = (scale_a1 * sBScale[scale_k, 1]).to(
                            self.final_acc_dtype
                        )
                        for acc_i in cutlass.range_constexpr(cute.size(tRS_rAcc)):
                            if cutlass.const_expr(acc_i < cute.size(tRS_rAcc) // 2):
                                scale = (
                                    scale_0_0
                                    if cutlass.const_expr(acc_i % 4 < 2)
                                    else scale_1_0
                                )
                            else:
                                scale = (
                                    scale_0_1
                                    if cutlass.const_expr(acc_i % 4 < 2)
                                    else scale_1_1
                                )
                            epi_idx = acc_i // size_tRS_rD
                            epi_v = acc_i - epi_idx * size_tRS_rD
                            contribution = (
                                tRS_rAcc[acc_i].to(self.final_acc_dtype) * scale
                            )
                            tRS_rFinal[epi_idx, epi_v] = contribution.to(
                                self.final_acc_dtype
                            )

                    for k_tile in range(1, k_tile_cnt // K_TILES_PER_SCALE_BLOCK):
                        accumulators.fill(0.0)
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                        cute.nvgpu.warpgroup.fence()
                        scale_k = (
                            mainloop_consumer_read_state.count
                            // K_TILES_PER_SCALE_BLOCK
                        )

                        for _ in cutlass.range_constexpr(K_TILES_PER_SCALE_BLOCK):
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_read_state
                            )
                            if cutlass.const_expr(self.wgrad):
                                scale_stage = mainloop_consumer_read_state.index
                                scale_a_0 = sAScale[0, scale_row_0, scale_stage]
                                scale_a_1 = sAScale[0, scale_row_1, scale_stage]
                                for scale_i in cutlass.range_constexpr(
                                    cute.size(tRS_rAcc) // 4
                                ):
                                    scale_n = scale_i * 8 + scale_col_pair
                                    scale_b_regs[scale_i * 2] = sBScale[
                                        0, scale_n, scale_stage
                                    ]
                                    scale_b_regs[scale_i * 2 + 1] = sBScale[
                                        0, scale_n + 1, scale_stage
                                    ]
                            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                                k_block_coord = (
                                    None,
                                    None,
                                    k_block_idx,
                                    mainloop_consumer_read_state.index,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    accumulators,
                                    tCrA[k_block_coord],
                                    tCrB[k_block_coord],
                                    accumulators,
                                )

                            cute.nvgpu.warpgroup.commit_group()
                            cute.nvgpu.warpgroup.wait_group(0)
                            mainloop_pipeline.consumer_release(
                                mainloop_consumer_release_state
                            )
                            mainloop_consumer_release_state.advance()
                            mainloop_consumer_read_state.advance()
                        if cutlass.const_expr(self.wgrad):
                            for acc_i in cutlass.range_constexpr(cute.size(tRS_rAcc)):
                                scale_i = acc_i // 4
                                scale_a = (
                                    scale_a_0
                                    if cutlass.const_expr(acc_i % 4 < 2)
                                    else scale_a_1
                                )
                                scale_b = scale_b_regs[scale_i * 2 + acc_i % 2]
                                epi_idx = acc_i // size_tRS_rD
                                epi_v = acc_i - epi_idx * size_tRS_rD
                                tRS_rFinal[epi_idx, epi_v] += (
                                    tRS_rAcc[acc_i] * scale_a * scale_b
                                )
                        else:
                            scale_a0 = sAScale[scale_buffer, scale_k, tRS_cS[0][0]]
                            scale_a1 = sAScale[scale_buffer, scale_k, tRS_cS[2][0]]
                            scale_0_0 = (scale_a0 * sBScale[scale_k, 0]).to(
                                self.final_acc_dtype
                            )
                            scale_1_0 = (scale_a1 * sBScale[scale_k, 0]).to(
                                self.final_acc_dtype
                            )
                            scale_0_1 = (scale_a0 * sBScale[scale_k, 1]).to(
                                self.final_acc_dtype
                            )
                            scale_1_1 = (scale_a1 * sBScale[scale_k, 1]).to(
                                self.final_acc_dtype
                            )
                            for acc_i in cutlass.range_constexpr(cute.size(tRS_rAcc)):
                                if cutlass.const_expr(acc_i < cute.size(tRS_rAcc) // 2):
                                    scale = (
                                        scale_0_0
                                        if cutlass.const_expr(acc_i % 4 < 2)
                                        else scale_1_0
                                    )
                                else:
                                    scale = (
                                        scale_0_1
                                        if cutlass.const_expr(acc_i % 4 < 2)
                                        else scale_1_1
                                    )
                                epi_idx = acc_i // size_tRS_rD
                                epi_v = acc_i - epi_idx * size_tRS_rD
                                contribution = (
                                    tRS_rAcc[acc_i].to(self.final_acc_dtype) * scale
                                )
                                final = tRS_rFinal[epi_idx, epi_v] + contribution
                                tRS_rFinal[epi_idx, epi_v] = final.to(
                                    self.final_acc_dtype
                                )

                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        tCgC_for_tma_partition,
                    )
                    num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                    for epi_idx in cutlass.range_constexpr(epi_tile_num):
                        for epi_v in cutlass.range_constexpr(size_tRS_rD):
                            tRS_rD[epi_v] = tRS_rFinal[epi_idx, epi_v]

                        acc_vec = tRS_rD.load()
                        tRS_rD_out.store(acc_vec.to(self.c_dtype))

                        epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                        if warp_idx == self.epi_store_warp_id:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, gmem_coord)],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()

                        self.epilog_sync_barrier.arrive_and_wait()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                tma_store_pipeline.producer_tail()

    class HopperWgmmaBlockwiseScaledPersistentWgradGemmKernel(
        HopperWgmmaBlockwiseScaledPersistentDgradGemmKernel
    ):
        pass

    kernel_cls = (
        HopperWgmmaBlockwiseScaledPersistentWgradGemmKernel
        if wgrad
        else HopperWgmmaBlockwiseScaledPersistentDgradGemmKernel
        if scale_k_blocks == 16
        else HopperWgmmaBlockwiseScaledPersistentFwdGemmKernel
    )
    gemm = kernel_cls(
        cutlass.Float32,
        tile_shape_mn,
        cluster_shape_mn,
        swizzle_size,
        False,
        a_view.shape[0],
        scale_k_blocks,
        wgrad,
    )
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    compiled = cute.compile(
        gemm,
        mA,
        mB,
        mAS,
        mBS,
        mC,
        max_active_clusters,
        stream,
    )
    cached = (compiled, max_active_clusters)
    _HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED[key] = cached
    return cached


def _scale_accum_chunk_blocks(
    M: int,
    N: int,
    k_blocks: int,
    partial_dtype: torch.dtype,
) -> int:
    chunk_blocks = min(_SCALE_ACCUM_MAX_CHUNK_BLOCKS, k_blocks)
    use_huge_wide_split_partials = (
        partial_dtype == torch.bfloat16
        and N >= 4096
        and M * N >= 512 * 1024 * 1024
        and k_blocks % 4 == 0
    )
    use_dtype_sized_partials = partial_dtype == torch.bfloat16 and (
        N < 4096 or use_huge_wide_split_partials
    )
    if use_huge_wide_split_partials:
        chunk_blocks = min(chunk_blocks, 8)
    elif use_dtype_sized_partials and M * N >= 128 * 1024 * 1024:
        chunk_blocks = min(chunk_blocks, 8)
    partial_element_size = partial_dtype.itemsize if use_dtype_sized_partials else 4
    partial_bytes_per_k_block = M * N * partial_element_size
    max_partial_bytes = (
        _SCALE_ACCUM_HUGE_WIDE_SPLIT8_PARTIAL_BYTES
        if use_huge_wide_split_partials
        else _SCALE_ACCUM_MAX_PARTIAL_BYTES
    )
    while (
        chunk_blocks > 1
        and chunk_blocks * partial_bytes_per_k_block > max_partial_bytes
    ):
        chunk_blocks //= 2
    return chunk_blocks


def _can_use_direct_bf16_scale_output(M: int, N: int, k_blocks: int) -> bool:
    return (
        k_blocks <= _SCALE_ACCUM_MAX_CHUNK_BLOCKS
        and k_blocks * M * N * torch.bfloat16.itemsize
        <= _SCALE_OUTPUT_MAX_BF16_PARTIAL_BYTES
    )


def _hopper_dense_tile_shape_mn(N: int) -> tuple[int, int]:
    if N >= 4096:
        return _HOPPER_DENSE_WIDE_N_TILE_SHAPE_MN
    return _HOPPER_DENSE_TILE_SHAPE_MN


def _hopper_dense_persistent_cluster_shape_mn(
    tile_shape_mn: tuple[int, int],
) -> tuple[int, int]:
    if tile_shape_mn == (128, 128):
        return (2, 1)
    return (2, 2)


@functools.cache
def _compile_chunk_scale_accum_kernel(threads_per_block: int):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    class ScaleAccumKernel:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            M_PER_GROUP: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            idx = cutlass.Int64(bidx * THREADS_PER_BLOCK + tidx)
            total = M * N
            if idx < total:
                m = idx // N
                n = idx - m * N
                group = m // M_PER_GROUP
                n_block = n // BLOCK_SIZE
                prior = cutlass.Float32(0.0)
                if START_K_BLOCK != 0:
                    prior = cutlass.Float32(accum[m, n])
                value = prior
                for local_k in cutlass.range(0, CHUNK_BLOCKS, 1, unroll=1):
                    k_block = START_K_BLOCK + local_k
                    value += (
                        cutlass.Float32(partials[local_k, m, n])
                        * cutlass.Float32(a_s[m, k_block])
                        * cutlass.Float32(b_s[group, k_block, n_block])
                    )
                accum[m, n] = value

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            M_PER_GROUP: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                accum,
                a_s,
                b_s,
                M,
                N,
                M_PER_GROUP,
                START_K_BLOCK,
                CHUNK_BLOCKS,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
            ).launch(
                grid=(cute.ceil_div(M * N, threads_per_block), 1, 1),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    scale_chunk_blocks = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    partial_stride2 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()

    fake_partials = make_fake_tensor(
        cutlass.Float32,
        (scale_chunk_blocks, m, n),
        stride=(partial_stride0, partial_stride1, partial_stride2),
    )
    fake_accum = make_fake_tensor(
        cutlass.Float32,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        ScaleAccumKernel(),
        partials=fake_partials,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        M=0,
        N=0,
        M_PER_GROUP=0,
        START_K_BLOCK=0,
        CHUNK_BLOCKS=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_kernel(
    threads_per_block: int,
    m_per_group: int,
    partial_dtype_name: str,
    max_chunk_blocks: int,
    rows_per_cta: int,
    accum_aligned: bool,
    k_unroll: int,
    has_prior: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    PARTIAL_DTYPE = _cutlass_dtype(partial_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumKernel:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            ACCUM_ALIGNED: cutlass.Constexpr[bool],
            HAS_PRIOR: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            K_UNROLL: cutlass.Constexpr[int],
            MAX_CHUNK_BLOCKS: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_a_s = smem.allocate_tensor(
                cutlass.Float32, MAX_CHUNK_BLOCKS * ROWS_PER_CTA
            )
            shared_b_s = smem.allocate_tensor(
                cutlass.Float32, MAX_CHUNK_BLOCKS * SCALE_BLOCKS_PER_CTA
            )

            a_scale_loads = ROWS_PER_CTA
            b_scale_loads = MAX_CHUNK_BLOCKS * SCALE_BLOCKS_PER_CTA
            if tidx < a_scale_loads:
                row_i = tidx
                m = m_base + row_i
                for local_k in cutlass.range(0, CHUNK_BLOCKS, 1, unroll=K_UNROLL):
                    k_block = START_K_BLOCK + local_k
                    if STATIC_START_K_BLOCK >= 0:
                        k_block = STATIC_START_K_BLOCK + local_k
                    if ACCUM_ALIGNED:
                        shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k] = (
                            cutlass.Float32(a_s[m, k_block])
                        )
                    else:
                        if m < M:
                            shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k] = (
                                cutlass.Float32(a_s[m, k_block])
                            )
            if tidx >= a_scale_loads and tidx < a_scale_loads + b_scale_loads:
                b_load_idx = tidx - a_scale_loads
                local_k = b_load_idx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = b_load_idx - local_k * SCALE_BLOCKS_PER_CTA
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                n_block = n_block_base + scale_i_for_load
                if local_k < CHUNK_BLOCKS and n_block * BLOCK_SIZE < N:
                    shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i_for_load] = (
                        cutlass.Float32(b_s[group, k_block, n_block])
                    )

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                if ACCUM_ALIGNED:
                    prior = cutlass.Float32(0.0)
                    if HAS_PRIOR:
                        prior = cutlass.Float32(accum[m, n])
                    value = prior
                    for local_k in cutlass.range(0, CHUNK_BLOCKS, 1, unroll=K_UNROLL):
                        value += (
                            cutlass.Float32(partials[local_k, m, n])
                            * shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k]
                            * shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                    accum[m, n] = value
                else:
                    if m < M and n < N:
                        prior = cutlass.Float32(0.0)
                        if HAS_PRIOR:
                            prior = cutlass.Float32(accum[m, n])
                        value = prior
                        for local_k in cutlass.range(
                            0, CHUNK_BLOCKS, 1, unroll=K_UNROLL
                        ):
                            value += (
                                cutlass.Float32(partials[local_k, m, n])
                                * shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k]
                                * shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i]
                            )
                        accum[m, n] = value

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                accum,
                a_s,
                b_s,
                M,
                N,
                START_K_BLOCK,
                CHUNK_BLOCKS,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                ACCUM_ALIGNED=accum_aligned,
                HAS_PRIOR=has_prior,
                STATIC_START_K_BLOCK=static_start_k_block,
                K_UNROLL=k_unroll,
                MAX_CHUNK_BLOCKS=max_chunk_blocks,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block)
                    * cute.ceil_div(M, rows_per_cta),
                    1,
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    scale_chunk_blocks = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    partial_stride2 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()

    fake_partials = make_fake_tensor(
        PARTIAL_DTYPE,
        (scale_chunk_blocks, m, n),
        stride=(partial_stride0, partial_stride1, partial_stride2),
    )
    fake_accum = make_fake_tensor(
        cutlass.Float32,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumKernel(),
        partials=fake_partials,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        M=0,
        N=0,
        START_K_BLOCK=0,
        CHUNK_BLOCKS=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_output_kernel(
    threads_per_block: int,
    m_per_group: int,
    partial_dtype_name: str,
    c_dtype_name: str,
    max_chunk_blocks: int,
    rows_per_cta: int,
    output_aligned: bool,
    k_unroll: int,
    has_prior: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    PARTIAL_DTYPE = _cutlass_dtype(partial_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumOutputKernel:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            OUTPUT_ALIGNED: cutlass.Constexpr[bool],
            HAS_PRIOR: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            K_UNROLL: cutlass.Constexpr[int],
            MAX_CHUNK_BLOCKS: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_a_s = smem.allocate_tensor(
                cutlass.Float32, MAX_CHUNK_BLOCKS * ROWS_PER_CTA
            )
            shared_b_s = smem.allocate_tensor(
                cutlass.Float32, MAX_CHUNK_BLOCKS * SCALE_BLOCKS_PER_CTA
            )

            a_scale_loads = ROWS_PER_CTA
            b_scale_loads = MAX_CHUNK_BLOCKS * SCALE_BLOCKS_PER_CTA
            if tidx < a_scale_loads:
                row_i = tidx
                m = m_base + row_i
                for local_k in cutlass.range(0, CHUNK_BLOCKS, 1, unroll=K_UNROLL):
                    k_block = START_K_BLOCK + local_k
                    if STATIC_START_K_BLOCK >= 0:
                        k_block = STATIC_START_K_BLOCK + local_k
                    if OUTPUT_ALIGNED:
                        shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k] = (
                            cutlass.Float32(a_s[m, k_block])
                        )
                    else:
                        if m < M:
                            shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k] = (
                                cutlass.Float32(a_s[m, k_block])
                            )
            if tidx >= a_scale_loads and tidx < a_scale_loads + b_scale_loads:
                b_load_idx = tidx - a_scale_loads
                local_k = b_load_idx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = b_load_idx - local_k * SCALE_BLOCKS_PER_CTA
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                n_block = n_block_base + scale_i_for_load
                if local_k < CHUNK_BLOCKS and n_block * BLOCK_SIZE < N:
                    shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i_for_load] = (
                        cutlass.Float32(b_s[group, k_block, n_block])
                    )

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                if OUTPUT_ALIGNED:
                    prior = cutlass.Float32(0.0)
                    if HAS_PRIOR:
                        prior = cutlass.Float32(accum[m, n])
                    value = prior
                    for local_k in cutlass.range(0, CHUNK_BLOCKS, 1, unroll=K_UNROLL):
                        value += (
                            cutlass.Float32(partials[local_k, m, n])
                            * shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k]
                            * shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                    out[m, n] = C_DTYPE(value)
                else:
                    if m < M and n < N:
                        prior = cutlass.Float32(0.0)
                        if HAS_PRIOR:
                            prior = cutlass.Float32(accum[m, n])
                        value = prior
                        for local_k in cutlass.range(
                            0, CHUNK_BLOCKS, 1, unroll=K_UNROLL
                        ):
                            value += (
                                cutlass.Float32(partials[local_k, m, n])
                                * shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k]
                                * shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i]
                            )
                        out[m, n] = C_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                accum,
                a_s,
                b_s,
                out,
                M,
                N,
                START_K_BLOCK,
                CHUNK_BLOCKS,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                OUTPUT_ALIGNED=output_aligned,
                HAS_PRIOR=has_prior,
                STATIC_START_K_BLOCK=static_start_k_block,
                K_UNROLL=k_unroll,
                MAX_CHUNK_BLOCKS=max_chunk_blocks,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block)
                    * cute.ceil_div(M, rows_per_cta),
                    1,
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    scale_chunk_blocks = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    partial_stride2 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()

    fake_partials = make_fake_tensor(
        PARTIAL_DTYPE,
        (scale_chunk_blocks, m, n),
        stride=(partial_stride0, partial_stride1, partial_stride2),
    )
    fake_accum = make_fake_tensor(
        cutlass.Float32,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_out = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(out_stride0, out_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumOutputKernel(),
        partials=fake_partials,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        M=0,
        N=0,
        START_K_BLOCK=0,
        CHUNK_BLOCKS=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_contig8_kernel(
    threads_per_block: int,
    m_per_group: int,
    partial_dtype_name: str,
    accum_dtype_name: str,
    rows_per_cta: int,
    accum_aligned: bool,
    has_prior: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    PARTIAL_DTYPE = _cutlass_dtype(partial_dtype_name)
    ACCUM_DTYPE = _cutlass_dtype(accum_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumContig8Kernel:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            ACCUM_ALIGNED: cutlass.Constexpr[bool],
            HAS_PRIOR: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 8
                local_k = row_k_i - row_i * 8
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if ACCUM_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, k_block]
                    ) * cutlass.Float32(b_s[group, k_block, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, k_block]
                        ) * cutlass.Float32(b_s[group, k_block, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 8 * SCALE_BLOCKS_PER_CTA
                if ACCUM_ALIGNED:
                    value = cutlass.Float32(0.0)
                    if HAS_PRIOR:
                        value = cutlass.Float32(accum[m, n])
                    for local_k in cutlass.range_constexpr(8):
                        value += (
                            cutlass.Float32(partials[local_k, m, n])
                            * shared_scale[
                                scale_base + local_k * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                    accum[m, n] = ACCUM_DTYPE(value)
                else:
                    if m < M and n < N:
                        value = cutlass.Float32(0.0)
                        if HAS_PRIOR:
                            value = cutlass.Float32(accum[m, n])
                        for local_k in cutlass.range_constexpr(8):
                            value += (
                                cutlass.Float32(partials[local_k, m, n])
                                * shared_scale[
                                    scale_base
                                    + local_k * SCALE_BLOCKS_PER_CTA
                                    + scale_i
                                ]
                            )
                        accum[m, n] = ACCUM_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                accum,
                a_s,
                b_s,
                M,
                N,
                START_K_BLOCK,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                ACCUM_ALIGNED=accum_aligned,
                HAS_PRIOR=has_prior,
                STATIC_START_K_BLOCK=static_start_k_block,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block)
                    * cute.ceil_div(M, rows_per_cta),
                    1,
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    scale_chunk_blocks = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    partial_stride2 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()

    fake_partials = make_fake_tensor(
        PARTIAL_DTYPE,
        (scale_chunk_blocks, m, n),
        stride=(partial_stride0, partial_stride1, partial_stride2),
    )
    fake_accum = make_fake_tensor(
        ACCUM_DTYPE,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumContig8Kernel(),
        partials=fake_partials,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        M=0,
        N=0,
        START_K_BLOCK=0,
        CHUNK_BLOCKS=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_output_contig8_kernel(
    threads_per_block: int,
    m_per_group: int,
    partial_dtype_name: str,
    accum_dtype_name: str,
    c_dtype_name: str,
    rows_per_cta: int,
    output_aligned: bool,
    has_prior: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    PARTIAL_DTYPE = _cutlass_dtype(partial_dtype_name)
    ACCUM_DTYPE = _cutlass_dtype(accum_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumOutputContig8Kernel:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            OUTPUT_ALIGNED: cutlass.Constexpr[bool],
            HAS_PRIOR: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 8
                local_k = row_k_i - row_i * 8
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if OUTPUT_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, k_block]
                    ) * cutlass.Float32(b_s[group, k_block, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, k_block]
                        ) * cutlass.Float32(b_s[group, k_block, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 8 * SCALE_BLOCKS_PER_CTA
                if OUTPUT_ALIGNED:
                    value = cutlass.Float32(0.0)
                    if HAS_PRIOR:
                        value = cutlass.Float32(accum[m, n])
                    for local_k in cutlass.range_constexpr(8):
                        value += (
                            cutlass.Float32(partials[local_k, m, n])
                            * shared_scale[
                                scale_base + local_k * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                    out[m, n] = C_DTYPE(value)
                else:
                    if m < M and n < N:
                        value = cutlass.Float32(0.0)
                        if HAS_PRIOR:
                            value = cutlass.Float32(accum[m, n])
                        for local_k in cutlass.range_constexpr(8):
                            value += (
                                cutlass.Float32(partials[local_k, m, n])
                                * shared_scale[
                                    scale_base
                                    + local_k * SCALE_BLOCKS_PER_CTA
                                    + scale_i
                                ]
                            )
                        out[m, n] = C_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            CHUNK_BLOCKS: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                accum,
                a_s,
                b_s,
                out,
                M,
                N,
                START_K_BLOCK,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                OUTPUT_ALIGNED=output_aligned,
                HAS_PRIOR=has_prior,
                STATIC_START_K_BLOCK=static_start_k_block,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block)
                    * cute.ceil_div(M, rows_per_cta),
                    1,
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    scale_chunk_blocks = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    partial_stride2 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()

    fake_partials = make_fake_tensor(
        PARTIAL_DTYPE,
        (scale_chunk_blocks, m, n),
        stride=(partial_stride0, partial_stride1, partial_stride2),
    )
    fake_accum = make_fake_tensor(
        ACCUM_DTYPE,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_out = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(out_stride0, out_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumOutputContig8Kernel(),
        partials=fake_partials,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        M=0,
        N=0,
        START_K_BLOCK=0,
        CHUNK_BLOCKS=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_split4_kernel(
    threads_per_block: int,
    m_per_group: int,
    accum_dtype_name: str,
    rows_per_cta: int,
    accum_aligned: bool,
    has_prior: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    ACCUM_DTYPE = _cutlass_dtype(accum_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumSplit4Kernel:
        @cute.kernel
        def kernel(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            ACCUM_ALIGNED: cutlass.Constexpr[bool],
            HAS_PRIOR: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            n_tile, m_tile, _ = cute.arch.block_idx()
            n_tile = cutlass.Int64(n_tile)
            m_tile = cutlass.Int64(m_tile)
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 4 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 4 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 4
                local_k = row_k_i - row_i * 4
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if ACCUM_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, k_block]
                    ) * cutlass.Float32(b_s[group, k_block, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, k_block]
                        ) * cutlass.Float32(b_s[group, k_block, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 4 * SCALE_BLOCKS_PER_CTA
                if ACCUM_ALIGNED:
                    prior = cutlass.Float32(0.0)
                    if HAS_PRIOR:
                        prior = cutlass.Float32(accum[m, n])
                    value = prior
                    value += (
                        cutlass.Float32(partial0[m, n])
                        * shared_scale[scale_base + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial1[m, n])
                        * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial2[m, n])
                        * shared_scale[scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial3[m, n])
                        * shared_scale[scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    accum[m, n] = ACCUM_DTYPE(value)
                else:
                    if m < M and n < N:
                        prior = cutlass.Float32(0.0)
                        if HAS_PRIOR:
                            prior = cutlass.Float32(accum[m, n])
                        value = prior
                        value += (
                            cutlass.Float32(partial0[m, n])
                            * shared_scale[scale_base + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial1[m, n])
                            * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial2[m, n])
                            * shared_scale[
                                scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial3[m, n])
                            * shared_scale[
                                scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        accum[m, n] = ACCUM_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial0,
                partial1,
                partial2,
                partial3,
                accum,
                a_s,
                b_s,
                M,
                N,
                START_K_BLOCK,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                ACCUM_ALIGNED=accum_aligned,
                HAS_PRIOR=has_prior,
                STATIC_START_K_BLOCK=static_start_k_block,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block),
                    cute.ceil_div(M, rows_per_cta),
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()

    fake_partial = make_fake_tensor(
        cutlass.BFloat16,
        (m, n),
        stride=(partial_stride0, partial_stride1),
    )
    fake_accum = make_fake_tensor(
        ACCUM_DTYPE,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumSplit4Kernel(),
        partial0=fake_partial,
        partial1=fake_partial,
        partial2=fake_partial,
        partial3=fake_partial,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        M=0,
        N=0,
        START_K_BLOCK=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_output_split4_kernel(
    threads_per_block: int,
    m_per_group: int,
    accum_dtype_name: str,
    c_dtype_name: str,
    rows_per_cta: int,
    output_aligned: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    ACCUM_DTYPE = _cutlass_dtype(accum_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumOutputSplit4Kernel:
        @cute.kernel
        def kernel(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            OUTPUT_ALIGNED: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            n_tile, m_tile, _ = cute.arch.block_idx()
            n_tile = cutlass.Int64(n_tile)
            m_tile = cutlass.Int64(m_tile)
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 4 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 4 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 4
                local_k = row_k_i - row_i * 4
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if OUTPUT_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, k_block]
                    ) * cutlass.Float32(b_s[group, k_block, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, k_block]
                        ) * cutlass.Float32(b_s[group, k_block, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 4 * SCALE_BLOCKS_PER_CTA
                if OUTPUT_ALIGNED:
                    prior = cutlass.Float32(accum[m, n])
                    value = prior
                    value += (
                        cutlass.Float32(partial0[m, n])
                        * shared_scale[scale_base + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial1[m, n])
                        * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial2[m, n])
                        * shared_scale[scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial3[m, n])
                        * shared_scale[scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    out[m, n] = C_DTYPE(value)
                else:
                    if m < M and n < N:
                        prior = cutlass.Float32(accum[m, n])
                        value = prior
                        value += (
                            cutlass.Float32(partial0[m, n])
                            * shared_scale[scale_base + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial1[m, n])
                            * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial2[m, n])
                            * shared_scale[
                                scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial3[m, n])
                            * shared_scale[
                                scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        out[m, n] = C_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial0,
                partial1,
                partial2,
                partial3,
                accum,
                a_s,
                b_s,
                out,
                M,
                N,
                START_K_BLOCK,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                OUTPUT_ALIGNED=output_aligned,
                STATIC_START_K_BLOCK=static_start_k_block,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block),
                    cute.ceil_div(M, rows_per_cta),
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()

    fake_partial = make_fake_tensor(
        cutlass.BFloat16,
        (m, n),
        stride=(partial_stride0, partial_stride1),
    )
    fake_accum = make_fake_tensor(
        ACCUM_DTYPE,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_out = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(out_stride0, out_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumOutputSplit4Kernel(),
        partial0=fake_partial,
        partial1=fake_partial,
        partial2=fake_partial,
        partial3=fake_partial,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        M=0,
        N=0,
        START_K_BLOCK=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_split8_kernel(
    threads_per_block: int,
    m_per_group: int,
    accum_dtype_name: str,
    rows_per_cta: int,
    accum_aligned: bool,
    has_prior: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    ACCUM_DTYPE = _cutlass_dtype(accum_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumSplit8Kernel:
        @cute.kernel
        def kernel(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            partial4: cute.Tensor,
            partial5: cute.Tensor,
            partial6: cute.Tensor,
            partial7: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            ACCUM_ALIGNED: cutlass.Constexpr[bool],
            HAS_PRIOR: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            n_tile, m_tile, _ = cute.arch.block_idx()
            n_tile = cutlass.Int64(n_tile)
            m_tile = cutlass.Int64(m_tile)
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 8
                local_k = row_k_i - row_i * 8
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if ACCUM_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, k_block]
                    ) * cutlass.Float32(b_s[group, k_block, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, k_block]
                        ) * cutlass.Float32(b_s[group, k_block, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 8 * SCALE_BLOCKS_PER_CTA
                if ACCUM_ALIGNED:
                    prior = cutlass.Float32(0.0)
                    if HAS_PRIOR:
                        prior = cutlass.Float32(accum[m, n])
                    value = prior
                    value += (
                        cutlass.Float32(partial0[m, n])
                        * shared_scale[scale_base + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial1[m, n])
                        * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial2[m, n])
                        * shared_scale[scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial3[m, n])
                        * shared_scale[scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial4[m, n])
                        * shared_scale[scale_base + 4 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial5[m, n])
                        * shared_scale[scale_base + 5 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial6[m, n])
                        * shared_scale[scale_base + 6 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial7[m, n])
                        * shared_scale[scale_base + 7 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    accum[m, n] = ACCUM_DTYPE(value)
                else:
                    if m < M and n < N:
                        prior = cutlass.Float32(0.0)
                        if HAS_PRIOR:
                            prior = cutlass.Float32(accum[m, n])
                        value = prior
                        value += (
                            cutlass.Float32(partial0[m, n])
                            * shared_scale[scale_base + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial1[m, n])
                            * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial2[m, n])
                            * shared_scale[
                                scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial3[m, n])
                            * shared_scale[
                                scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial4[m, n])
                            * shared_scale[
                                scale_base + 4 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial5[m, n])
                            * shared_scale[
                                scale_base + 5 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial6[m, n])
                            * shared_scale[
                                scale_base + 6 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial7[m, n])
                            * shared_scale[
                                scale_base + 7 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        accum[m, n] = ACCUM_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            partial4: cute.Tensor,
            partial5: cute.Tensor,
            partial6: cute.Tensor,
            partial7: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial0,
                partial1,
                partial2,
                partial3,
                partial4,
                partial5,
                partial6,
                partial7,
                accum,
                a_s,
                b_s,
                M,
                N,
                START_K_BLOCK,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                ACCUM_ALIGNED=accum_aligned,
                HAS_PRIOR=has_prior,
                STATIC_START_K_BLOCK=static_start_k_block,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block),
                    cute.ceil_div(M, rows_per_cta),
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()

    fake_partial = make_fake_tensor(
        cutlass.BFloat16,
        (m, n),
        stride=(partial_stride0, partial_stride1),
    )
    fake_accum = make_fake_tensor(
        ACCUM_DTYPE,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumSplit8Kernel(),
        partial0=fake_partial,
        partial1=fake_partial,
        partial2=fake_partial,
        partial3=fake_partial,
        partial4=fake_partial,
        partial5=fake_partial,
        partial6=fake_partial,
        partial7=fake_partial,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        M=0,
        N=0,
        START_K_BLOCK=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_accum_output_split8_kernel(
    threads_per_block: int,
    m_per_group: int,
    accum_dtype_name: str,
    c_dtype_name: str,
    rows_per_cta: int,
    output_aligned: bool,
    static_start_k_block: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    ACCUM_DTYPE = _cutlass_dtype(accum_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleAccumOutputSplit8Kernel:
        @cute.kernel
        def kernel(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            partial4: cute.Tensor,
            partial5: cute.Tensor,
            partial6: cute.Tensor,
            partial7: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            OUTPUT_ALIGNED: cutlass.Constexpr[bool],
            STATIC_START_K_BLOCK: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            n_tile, m_tile, _ = cute.arch.block_idx()
            n_tile = cutlass.Int64(n_tile)
            m_tile = cutlass.Int64(m_tile)
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 8 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 8
                local_k = row_k_i - row_i * 8
                k_block = START_K_BLOCK + local_k
                if STATIC_START_K_BLOCK >= 0:
                    k_block = STATIC_START_K_BLOCK + local_k
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if OUTPUT_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, k_block]
                    ) * cutlass.Float32(b_s[group, k_block, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, k_block]
                        ) * cutlass.Float32(b_s[group, k_block, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 8 * SCALE_BLOCKS_PER_CTA
                if OUTPUT_ALIGNED:
                    value = cutlass.Float32(accum[m, n])
                    value += (
                        cutlass.Float32(partial0[m, n])
                        * shared_scale[scale_base + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial1[m, n])
                        * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial2[m, n])
                        * shared_scale[scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial3[m, n])
                        * shared_scale[scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial4[m, n])
                        * shared_scale[scale_base + 4 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial5[m, n])
                        * shared_scale[scale_base + 5 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial6[m, n])
                        * shared_scale[scale_base + 6 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial7[m, n])
                        * shared_scale[scale_base + 7 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    out[m, n] = C_DTYPE(value)
                else:
                    if m < M and n < N:
                        value = cutlass.Float32(accum[m, n])
                        value += (
                            cutlass.Float32(partial0[m, n])
                            * shared_scale[scale_base + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial1[m, n])
                            * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial2[m, n])
                            * shared_scale[
                                scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial3[m, n])
                            * shared_scale[
                                scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial4[m, n])
                            * shared_scale[
                                scale_base + 4 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial5[m, n])
                            * shared_scale[
                                scale_base + 5 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial6[m, n])
                            * shared_scale[
                                scale_base + 6 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial7[m, n])
                            * shared_scale[
                                scale_base + 7 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        out[m, n] = C_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            partial4: cute.Tensor,
            partial5: cute.Tensor,
            partial6: cute.Tensor,
            partial7: cute.Tensor,
            accum: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            START_K_BLOCK: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial0,
                partial1,
                partial2,
                partial3,
                partial4,
                partial5,
                partial6,
                partial7,
                accum,
                a_s,
                b_s,
                out,
                M,
                N,
                START_K_BLOCK,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                OUTPUT_ALIGNED=output_aligned,
                STATIC_START_K_BLOCK=static_start_k_block,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block),
                    cute.ceil_div(M, rows_per_cta),
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    accum_stride0 = cute.sym_int()
    accum_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()

    fake_partial = make_fake_tensor(
        cutlass.BFloat16,
        (m, n),
        stride=(partial_stride0, partial_stride1),
    )
    fake_accum = make_fake_tensor(
        ACCUM_DTYPE,
        (m, n),
        stride=(accum_stride0, accum_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_out = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(out_stride0, out_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleAccumOutputSplit8Kernel(),
        partial0=fake_partial,
        partial1=fake_partial,
        partial2=fake_partial,
        partial3=fake_partial,
        partial4=fake_partial,
        partial5=fake_partial,
        partial6=fake_partial,
        partial7=fake_partial,
        accum=fake_accum,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        M=0,
        N=0,
        START_K_BLOCK=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_output_split16_kernel(
    threads_per_block: int,
    m_per_group: int,
    c_dtype_name: str,
    rows_per_cta: int,
    output_aligned: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    C_DTYPE = _cutlass_dtype(c_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleOutputSplit16Kernel:
        @cute.kernel
        def kernel(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            partial4: cute.Tensor,
            partial5: cute.Tensor,
            partial6: cute.Tensor,
            partial7: cute.Tensor,
            partial8: cute.Tensor,
            partial9: cute.Tensor,
            partial10: cute.Tensor,
            partial11: cute.Tensor,
            partial12: cute.Tensor,
            partial13: cute.Tensor,
            partial14: cute.Tensor,
            partial15: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            OUTPUT_ALIGNED: cutlass.Constexpr[bool],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_scale = smem.allocate_tensor(
                cutlass.Float32, 16 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            )

            scale_loads = 16 * ROWS_PER_CTA * SCALE_BLOCKS_PER_CTA
            if tidx < scale_loads:
                row_k_i = tidx // SCALE_BLOCKS_PER_CTA
                scale_i_for_load = tidx - row_k_i * SCALE_BLOCKS_PER_CTA
                row_i = row_k_i // 16
                local_k = row_k_i - row_i * 16
                m = m_base + row_i
                n_block = n_block_base + scale_i_for_load
                if OUTPUT_ALIGNED:
                    shared_scale[tidx] = cutlass.Float32(
                        a_s[m, local_k]
                    ) * cutlass.Float32(b_s[group, local_k, n_block])
                else:
                    if m < M and n_block * BLOCK_SIZE < N:
                        shared_scale[tidx] = cutlass.Float32(
                            a_s[m, local_k]
                        ) * cutlass.Float32(b_s[group, local_k, n_block])

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                scale_base = row_i * 16 * SCALE_BLOCKS_PER_CTA
                if OUTPUT_ALIGNED:
                    value = (
                        cutlass.Float32(partial0[m, n])
                        * shared_scale[scale_base + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial1[m, n])
                        * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial2[m, n])
                        * shared_scale[scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial3[m, n])
                        * shared_scale[scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial4[m, n])
                        * shared_scale[scale_base + 4 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial5[m, n])
                        * shared_scale[scale_base + 5 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial6[m, n])
                        * shared_scale[scale_base + 6 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial7[m, n])
                        * shared_scale[scale_base + 7 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial8[m, n])
                        * shared_scale[scale_base + 8 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial9[m, n])
                        * shared_scale[scale_base + 9 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial10[m, n])
                        * shared_scale[scale_base + 10 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial11[m, n])
                        * shared_scale[scale_base + 11 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial12[m, n])
                        * shared_scale[scale_base + 12 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial13[m, n])
                        * shared_scale[scale_base + 13 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial14[m, n])
                        * shared_scale[scale_base + 14 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    value += (
                        cutlass.Float32(partial15[m, n])
                        * shared_scale[scale_base + 15 * SCALE_BLOCKS_PER_CTA + scale_i]
                    )
                    out[m, n] = C_DTYPE(value)
                else:
                    if m < M and n < N:
                        value = (
                            cutlass.Float32(partial0[m, n])
                            * shared_scale[scale_base + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial1[m, n])
                            * shared_scale[scale_base + SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                        value += (
                            cutlass.Float32(partial2[m, n])
                            * shared_scale[
                                scale_base + 2 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial3[m, n])
                            * shared_scale[
                                scale_base + 3 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial4[m, n])
                            * shared_scale[
                                scale_base + 4 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial5[m, n])
                            * shared_scale[
                                scale_base + 5 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial6[m, n])
                            * shared_scale[
                                scale_base + 6 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial7[m, n])
                            * shared_scale[
                                scale_base + 7 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial8[m, n])
                            * shared_scale[
                                scale_base + 8 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial9[m, n])
                            * shared_scale[
                                scale_base + 9 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial10[m, n])
                            * shared_scale[
                                scale_base + 10 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial11[m, n])
                            * shared_scale[
                                scale_base + 11 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial12[m, n])
                            * shared_scale[
                                scale_base + 12 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial13[m, n])
                            * shared_scale[
                                scale_base + 13 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial14[m, n])
                            * shared_scale[
                                scale_base + 14 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        value += (
                            cutlass.Float32(partial15[m, n])
                            * shared_scale[
                                scale_base + 15 * SCALE_BLOCKS_PER_CTA + scale_i
                            ]
                        )
                        out[m, n] = C_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partial0: cute.Tensor,
            partial1: cute.Tensor,
            partial2: cute.Tensor,
            partial3: cute.Tensor,
            partial4: cute.Tensor,
            partial5: cute.Tensor,
            partial6: cute.Tensor,
            partial7: cute.Tensor,
            partial8: cute.Tensor,
            partial9: cute.Tensor,
            partial10: cute.Tensor,
            partial11: cute.Tensor,
            partial12: cute.Tensor,
            partial13: cute.Tensor,
            partial14: cute.Tensor,
            partial15: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partial0,
                partial1,
                partial2,
                partial3,
                partial4,
                partial5,
                partial6,
                partial7,
                partial8,
                partial9,
                partial10,
                partial11,
                partial12,
                partial13,
                partial14,
                partial15,
                a_s,
                b_s,
                out,
                M,
                N,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                OUTPUT_ALIGNED=output_aligned,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block)
                    * cute.ceil_div(M, rows_per_cta),
                    1,
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()

    fake_partial = make_fake_tensor(
        cutlass.BFloat16,
        (m, n),
        stride=(partial_stride0, partial_stride1),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_out = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(out_stride0, out_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleOutputSplit16Kernel(),
        partial0=fake_partial,
        partial1=fake_partial,
        partial2=fake_partial,
        partial3=fake_partial,
        partial4=fake_partial,
        partial5=fake_partial,
        partial6=fake_partial,
        partial7=fake_partial,
        partial8=fake_partial,
        partial9=fake_partial,
        partial10=fake_partial,
        partial11=fake_partial,
        partial12=fake_partial,
        partial13=fake_partial,
        partial14=fake_partial,
        partial15=fake_partial,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        M=0,
        N=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_shared_scale_output_kernel(
    threads_per_block: int,
    m_per_group: int,
    partial_dtype_name: str,
    c_dtype_name: str,
    chunk_blocks: int,
    rows_per_cta: int,
    k_unroll: int,
    output_aligned: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    PARTIAL_DTYPE = _cutlass_dtype(partial_dtype_name)
    C_DTYPE = _cutlass_dtype(c_dtype_name)
    scale_blocks_per_cta = threads_per_block // 128

    class SharedScaleOutputKernel:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            CHUNK_BLOCKS: cutlass.Constexpr[int],
            M_PER_GROUP: cutlass.Constexpr[int],
            BLOCK_SIZE: cutlass.Constexpr[int],
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
            ROWS_PER_CTA: cutlass.Constexpr[int],
            OUTPUT_ALIGNED: cutlass.Constexpr[bool],
            MAX_CHUNK_BLOCKS: cutlass.Constexpr[int],
            SCALE_BLOCKS_PER_CTA: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            n_tiles = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            m_tile = cta_idx // n_tiles
            n_tile = cta_idx - m_tile * n_tiles
            m_base = cutlass.Int64(m_tile * ROWS_PER_CTA)
            n = cutlass.Int64(n_tile * THREADS_PER_BLOCK + tidx)
            group = m_base // M_PER_GROUP
            n_block_base = n_tile * SCALE_BLOCKS_PER_CTA

            smem = cutlass.utils.SmemAllocator()
            shared_a_s = smem.allocate_tensor(
                cutlass.Float32, MAX_CHUNK_BLOCKS * ROWS_PER_CTA
            )
            shared_b_s = smem.allocate_tensor(
                cutlass.Float32, MAX_CHUNK_BLOCKS * SCALE_BLOCKS_PER_CTA
            )

            if tidx < CHUNK_BLOCKS:
                local_k = tidx
                k_block = local_k
                for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                    m = m_base + row_i
                    if OUTPUT_ALIGNED:
                        shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k] = (
                            cutlass.Float32(a_s[m, k_block])
                        )
                    else:
                        if m < M:
                            shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k] = (
                                cutlass.Float32(a_s[m, k_block])
                            )
                for scale_i in cutlass.range_constexpr(SCALE_BLOCKS_PER_CTA):
                    n_block = n_block_base + scale_i
                    if OUTPUT_ALIGNED:
                        shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i] = (
                            cutlass.Float32(b_s[group, k_block, n_block])
                        )
                    else:
                        if n_block * BLOCK_SIZE < N:
                            shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i] = (
                                cutlass.Float32(b_s[group, k_block, n_block])
                            )

            cute.arch.sync_threads()

            scale_i = tidx // BLOCK_SIZE
            for row_i in cutlass.range_constexpr(ROWS_PER_CTA):
                m = m_base + row_i
                if OUTPUT_ALIGNED:
                    value = cutlass.Float32(0.0)
                    for local_k in cutlass.range(0, CHUNK_BLOCKS, 1, unroll=k_unroll):
                        value += (
                            cutlass.Float32(partials[local_k, m, n])
                            * shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k]
                            * shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i]
                        )
                    out[m, n] = C_DTYPE(value)
                else:
                    if m < M and n < N:
                        value = cutlass.Float32(0.0)
                        for local_k in cutlass.range(
                            0, CHUNK_BLOCKS, 1, unroll=k_unroll
                        ):
                            value += (
                                cutlass.Float32(partials[local_k, m, n])
                                * shared_a_s[row_i * MAX_CHUNK_BLOCKS + local_k]
                                * shared_b_s[local_k * SCALE_BLOCKS_PER_CTA + scale_i]
                            )
                        out[m, n] = C_DTYPE(value)

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            M: cutlass.Int64,
            N: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                a_s,
                b_s,
                out,
                M,
                N,
                CHUNK_BLOCKS=chunk_blocks,
                M_PER_GROUP=m_per_group,
                BLOCK_SIZE=128,
                THREADS_PER_BLOCK=threads_per_block,
                ROWS_PER_CTA=rows_per_cta,
                OUTPUT_ALIGNED=output_aligned,
                MAX_CHUNK_BLOCKS=chunk_blocks,
                SCALE_BLOCKS_PER_CTA=scale_blocks_per_cta,
            ).launch(
                grid=(
                    cute.ceil_div(N, threads_per_block)
                    * cute.ceil_div(M, rows_per_cta),
                    1,
                    1,
                ),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    m = cute.sym_int()
    n = cute.sym_int()
    e = cute.sym_int()
    scale_chunk_blocks = cute.sym_int()
    k_blocks = cute.sym_int()
    n_blocks = cute.sym_int()
    padded_k_blocks = cute.sym_int()
    partial_stride0 = cute.sym_int()
    partial_stride1 = cute.sym_int()
    partial_stride2 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    b_s_stride2 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()

    fake_partials = make_fake_tensor(
        PARTIAL_DTYPE,
        (scale_chunk_blocks, m, n),
        stride=(partial_stride0, partial_stride1, partial_stride2),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (m, k_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (e, padded_k_blocks, n_blocks),
        stride=(b_s_stride0, b_s_stride1, b_s_stride2),
    )
    fake_out = make_fake_tensor(
        C_DTYPE,
        (m, n),
        stride=(out_stride0, out_stride1),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        SharedScaleOutputKernel(),
        partials=fake_partials,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        M=0,
        N=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def _cutedsl_splitk_equal_group_scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor | None:
    if out_dtype not in (torch.bfloat16, torch.float32):
        return None
    if block_size != 128 or not _hopper_dense_gemm_available():
        return None

    M, K = a.shape
    E, b_k, N = b.shape
    if K != b_k or K % block_size != 0 or N % block_size != 0:
        return None
    M_per_group = _equal_group_size_from_offsets(offs, M)
    if M_per_group is None:
        return None

    k_blocks = K // block_size
    partial_torch_dtype = (
        torch.bfloat16 if out_dtype == torch.bfloat16 else torch.float32
    )
    use_direct_scale_output = (
        out_dtype == torch.bfloat16
        and _can_use_direct_bf16_scale_output(M, N, k_blocks)
    )
    use_split16_direct_scale_output = (
        out_dtype == torch.bfloat16
        and N >= 4096
        and k_blocks == 16
        and M_per_group % 8 == 0
        and k_blocks * M * N * torch.bfloat16.itemsize
        <= _SCALE_OUTPUT_SPLIT16_MAX_BF16_PARTIAL_BYTES
    )
    scale_chunk_blocks = _scale_accum_chunk_blocks(
        M,
        N,
        k_blocks,
        partial_torch_dtype,
    )
    if use_direct_scale_output:
        scale_chunk_blocks = k_blocks
    elif use_split16_direct_scale_output:
        scale_chunk_blocks = k_blocks
    elif out_dtype == torch.bfloat16 and scale_chunk_blocks == k_blocks:
        # Keep large BF16 outputs off the all-K partial path unless the direct
        # output workspace cap explicitly allows it.
        scale_chunk_blocks = max(1, scale_chunk_blocks // 2)
    tile_shape_mn = _hopper_dense_tile_shape_mn(N)
    use_persistent_dense = (
        out_dtype == torch.bfloat16
        and M_per_group >= 1024
        and E >= 4
        and tile_shape_mn in ((128, 128), (128, 256))
        and _cutedsl_runtime_available()
    )
    key = (M_per_group, N, E, block_size, tile_shape_mn)
    scale_threads_per_block = (
        512 if use_direct_scale_output and N >= 7168 and N % 512 == 0 else 256
    )
    use_shared_scale_accum = N % scale_threads_per_block == 0
    use_shared_scale_output = (
        use_shared_scale_accum
        and use_direct_scale_output
        and k_blocks == scale_chunk_blocks
    )

    import cutlass

    stream_id = int(torch.cuda.current_stream().cuda_stream)
    stream = _get_cuda_stream(stream_id)
    fused_tile_shape_mn = (
        _HOPPER_DENSE_FWD_TILE_SHAPE_MN
        if k_blocks == 56 and N >= _HOPPER_DENSE_FWD_TILE_SHAPE_MN[1]
        else _hopper_dense_tile_shape_mn(N)
    )
    use_fused_blockwise_scaled_dense = (
        out_dtype == torch.bfloat16
        and k_blocks in (16, 56)
        and M_per_group % fused_tile_shape_mn[0] == 0
        and (
            fused_tile_shape_mn == _HOPPER_DENSE_FWD_TILE_SHAPE_MN
            or N % fused_tile_shape_mn[1] == 0
        )
        and E >= 4
        and _cutedsl_runtime_available()
    )
    if use_fused_blockwise_scaled_dense:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        a_s_view = a_s.t()
        a_view = torch.as_strided(
            a,
            (M_per_group, K, E),
            (K, 1, M_per_group * K),
        )
        b_view = torch.as_strided(
            b,
            (N, K, E),
            (K, 1, K * N),
        )
        c_view = torch.as_strided(
            out,
            (M_per_group, N, E),
            (N, 1, M_per_group * N),
        )
        fused_key = (
            "blockwise_scaled_persistent",
            M_per_group,
            N,
            E,
            K,
            block_size,
            k_blocks,
            fused_tile_shape_mn,
            tuple(a_s_view.shape),
            tuple(a_s_view.stride()),
        )
        compiled_info = _get_hopper_blockwise_scaled_persistent_gemm(
            a_view,
            b_view,
            a_s_view,
            b_s,
            c_view,
            fused_key,
            fused_tile_shape_mn,
            k_blocks,
        )
        if compiled_info is not None:
            compiled, _ = compiled_info
            mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
            mB = _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN)
            mAS = _make_cutedsl_tensor(a_s_view, cutlass.Float32, leading_dim=1)
            mBS = _make_cutedsl_tensor(b_s, cutlass.Float32, leading_dim=1)
            mC = _make_hopper_dense_tensor(c_view, cutlass.BFloat16)
            compiled(mA, mB, mAS, mBS, mC, stream)
            return out

    use_bf16_partials = out_dtype == torch.bfloat16
    use_wrapped_scale_tensors = k_blocks > 16 and not use_shared_scale_output
    use_split_scale_partials = (
        out_dtype == torch.bfloat16
        and N >= 4096
        and scale_chunk_blocks in (4, 8, 16)
        and k_blocks % scale_chunk_blocks == 0
        and not use_shared_scale_output
    )
    use_kblock_batched_dense = (
        out_dtype == torch.bfloat16
        and not use_split_scale_partials
        and k_blocks >= 16
        and scale_chunk_blocks > 1
        and k_blocks % scale_chunk_blocks == 0
    )
    use_contig8_bf16_accum = (
        out_dtype == torch.bfloat16
        and not use_split_scale_partials
        and scale_chunk_blocks == 8
        and k_blocks % scale_chunk_blocks == 0
        and N < 4096
        and M * N >= 128 * 1024 * 1024
    )
    scale_accum_torch_dtype = (
        torch.bfloat16 if use_contig8_bf16_accum else torch.float32
    )
    scale_accum_dtype_name = _torch_dtype_name(scale_accum_torch_dtype)
    partial_cutlass_dtype = cutlass.BFloat16 if use_bf16_partials else cutlass.Float32
    partial_dtype_name = _torch_dtype_name(partial_torch_dtype)
    split_scale_accum_torch_dtype = (
        torch.bfloat16 if use_split_scale_partials else torch.float32
    )
    split_scale_accum_dtype_name = _torch_dtype_name(split_scale_accum_torch_dtype)
    if out_dtype == torch.bfloat16:
        if use_split_scale_partials:
            partials, accum, c_views, mCs = (
                _get_cached_splitk_bf16_split_partial_workspace(
                    M,
                    N,
                    E,
                    M_per_group,
                    scale_chunk_blocks,
                    not use_split16_direct_scale_output,
                    split_scale_accum_torch_dtype,
                    a.device,
                    stream_id,
                    partial_cutlass_dtype,
                )
            )
            batched_c_view = None
            batched_mC = None
            cached_scale_partials = None
        else:
            (
                partials,
                accum,
                c_views,
                mCs,
                batched_c_view,
                batched_mC,
                cached_scale_partials,
            ) = _get_cached_splitk_bf16_workspace(
                M,
                N,
                E,
                M_per_group,
                scale_chunk_blocks,
                not use_shared_scale_output,
                scale_accum_torch_dtype,
                partial_torch_dtype,
                a.device,
                stream_id,
                partial_cutlass_dtype,
                use_kblock_batched_dense,
                use_wrapped_scale_tensors,
            )
    else:
        accum = torch.empty((M, N), device=a.device, dtype=torch.float32)
        partials = torch.empty(
            (scale_chunk_blocks, M, N), device=a.device, dtype=torch.float32
        )
        c_views = [
            torch.as_strided(
                partials[local_k],
                (M_per_group, N, E),
                (N, 1, M_per_group * N),
            )
            for local_k in range(scale_chunk_blocks)
        ]
        mCs = [_make_hopper_dense_tensor(c_view, cutlass.Float32) for c_view in c_views]
        batched_c_view = None
        batched_mC = None
        cached_scale_partials = None
    mBs = None
    if not use_kblock_batched_dense:
        mBs = _get_cached_hopper_dense_b_kblock_tensors(
            b,
            block_size,
            cutlass.Float8E4M3FN,
        )
        if mBs is None:
            return None
    compiled = None
    scale_output = None
    scale_accum_output = None
    scale_accum_first = None
    scale_accum_static = None
    scale_accum = None
    out = None
    if use_shared_scale_output:
        scale_accum = None
    elif use_shared_scale_accum:
        if use_split_scale_partials and M_per_group % 8 == 0:
            scale_accum_rows_per_cta = 8
        elif (
            out_dtype == torch.bfloat16
            and not use_split_scale_partials
            and N < 4096
            and M * N >= 128 * 1024 * 1024
            and scale_chunk_blocks == 8
            and M_per_group % 4 == 0
        ):
            scale_accum_rows_per_cta = 4
        elif M_per_group % 16 == 0:
            scale_accum_rows_per_cta = 16
        elif M_per_group % 8 == 0:
            scale_accum_rows_per_cta = 8
        elif M_per_group % 4 == 0:
            scale_accum_rows_per_cta = 4
        elif M_per_group % 2 == 0:
            scale_accum_rows_per_cta = 2
        else:
            scale_accum_rows_per_cta = 1
        scale_accum_aligned = (
            M % scale_accum_rows_per_cta == 0 and N % scale_threads_per_block == 0
        )
        if use_split_scale_partials and use_split16_direct_scale_output:
            scale_accum_output_rows_per_cta = 2
        elif use_split_scale_partials and M_per_group % 8 == 0:
            scale_accum_output_rows_per_cta = 8
        elif (
            not use_split_scale_partials
            and N < 4096
            and M * N >= 128 * 1024 * 1024
            and M_per_group % 8 == 0
        ):
            scale_accum_output_rows_per_cta = 8
        else:
            scale_accum_output_rows_per_cta = scale_accum_rows_per_cta
        scale_accum_output_aligned = (
            M % scale_accum_output_rows_per_cta == 0
            and N % scale_threads_per_block == 0
        )
        use_large_forward_scale_unroll = N < 4096 and M * N >= 128 * 1024 * 1024
        scale_accum_k_unroll = 8 if use_large_forward_scale_unroll else 2
        use_contig8_scale_accum = (
            not use_split_scale_partials
            and out_dtype == torch.bfloat16
            and partial_torch_dtype == torch.bfloat16
            and scale_chunk_blocks == 8
            and k_blocks % scale_chunk_blocks == 0
        )
        if use_split_scale_partials and use_split16_direct_scale_output:
            scale_accum = None
        elif use_split_scale_partials:
            scale_accum_compile = (
                _compile_shared_scale_accum_split8_kernel
                if scale_chunk_blocks == 8
                else _compile_shared_scale_accum_split4_kernel
            )
            scale_accum_first = scale_accum_compile(
                scale_threads_per_block,
                int(M_per_group),
                split_scale_accum_dtype_name,
                int(scale_accum_rows_per_cta),
                bool(scale_accum_aligned),
                False,
                0,
            )
            scale_accum = scale_accum_compile(
                scale_threads_per_block,
                int(M_per_group),
                split_scale_accum_dtype_name,
                int(scale_accum_rows_per_cta),
                bool(scale_accum_aligned),
                True,
                -1,
            )
            scale_accum_static = {
                int(static_chunk_start): scale_accum_compile(
                    scale_threads_per_block,
                    int(M_per_group),
                    split_scale_accum_dtype_name,
                    int(scale_accum_rows_per_cta),
                    bool(scale_accum_aligned),
                    True,
                    int(static_chunk_start),
                )
                for static_chunk_start in range(
                    scale_chunk_blocks,
                    k_blocks - scale_chunk_blocks,
                    scale_chunk_blocks,
                )
            }
        else:
            if use_contig8_scale_accum:
                scale_accum_first = _compile_shared_scale_accum_contig8_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    partial_dtype_name,
                    scale_accum_dtype_name,
                    int(scale_accum_rows_per_cta),
                    bool(scale_accum_aligned),
                    False,
                    0,
                )
                scale_accum = _compile_shared_scale_accum_contig8_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    partial_dtype_name,
                    scale_accum_dtype_name,
                    int(scale_accum_rows_per_cta),
                    bool(scale_accum_aligned),
                    True,
                    -1,
                )
                scale_accum_static = {
                    int(static_chunk_start): _compile_shared_scale_accum_contig8_kernel(
                        scale_threads_per_block,
                        int(M_per_group),
                        partial_dtype_name,
                        scale_accum_dtype_name,
                        int(scale_accum_rows_per_cta),
                        bool(scale_accum_aligned),
                        True,
                        int(static_chunk_start),
                    )
                    for static_chunk_start in range(
                        scale_chunk_blocks,
                        k_blocks - scale_chunk_blocks,
                        scale_chunk_blocks,
                    )
                }
            else:
                scale_accum_first = _compile_shared_scale_accum_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    partial_dtype_name,
                    int(scale_chunk_blocks),
                    int(scale_accum_rows_per_cta),
                    bool(scale_accum_aligned),
                    int(scale_accum_k_unroll),
                    False,
                    0,
                )
                scale_accum = _compile_shared_scale_accum_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    partial_dtype_name,
                    int(scale_chunk_blocks),
                    int(scale_accum_rows_per_cta),
                    bool(scale_accum_aligned),
                    int(scale_accum_k_unroll),
                    True,
                    -1,
                )
                if use_large_forward_scale_unroll:
                    scale_accum_static = {
                        int(static_chunk_start): _compile_shared_scale_accum_kernel(
                            scale_threads_per_block,
                            int(M_per_group),
                            partial_dtype_name,
                            int(scale_chunk_blocks),
                            int(scale_accum_rows_per_cta),
                            bool(scale_accum_aligned),
                            int(scale_accum_k_unroll),
                            True,
                            int(static_chunk_start),
                        )
                        for static_chunk_start in range(
                            scale_chunk_blocks,
                            k_blocks - scale_chunk_blocks,
                            scale_chunk_blocks,
                        )
                    }
        if out_dtype == torch.bfloat16:
            out = torch.empty((M, N), device=a.device, dtype=out_dtype)
            if use_split_scale_partials and use_split16_direct_scale_output:
                scale_accum_output = _compile_shared_scale_output_split16_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    _torch_dtype_name(out_dtype),
                    int(scale_accum_output_rows_per_cta),
                    bool(scale_accum_output_aligned),
                )
            elif use_split_scale_partials:
                scale_accum_output_compile = (
                    _compile_shared_scale_accum_output_split8_kernel
                    if scale_chunk_blocks == 8
                    else _compile_shared_scale_accum_output_split4_kernel
                )
                scale_accum_output = scale_accum_output_compile(
                    scale_threads_per_block,
                    int(M_per_group),
                    split_scale_accum_dtype_name,
                    _torch_dtype_name(out_dtype),
                    int(scale_accum_output_rows_per_cta),
                    bool(scale_accum_output_aligned),
                    int(k_blocks - scale_chunk_blocks),
                )
            elif use_contig8_scale_accum:
                scale_accum_output = _compile_shared_scale_accum_output_contig8_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    partial_dtype_name,
                    scale_accum_dtype_name,
                    _torch_dtype_name(out_dtype),
                    int(scale_accum_output_rows_per_cta),
                    bool(scale_accum_output_aligned),
                    bool(k_blocks != scale_chunk_blocks),
                    int(k_blocks - scale_chunk_blocks),
                )
            else:
                scale_accum_output = _compile_shared_scale_accum_output_kernel(
                    scale_threads_per_block,
                    int(M_per_group),
                    partial_dtype_name,
                    _torch_dtype_name(out_dtype),
                    int(scale_chunk_blocks),
                    int(scale_accum_output_rows_per_cta),
                    bool(scale_accum_output_aligned),
                    int(scale_accum_k_unroll),
                    bool(k_blocks != scale_chunk_blocks),
                    int(k_blocks - scale_chunk_blocks),
                )
    else:
        scale_accum = _compile_chunk_scale_accum_kernel(scale_threads_per_block)
    if use_shared_scale_output:
        out = torch.empty((M, N), device=a.device, dtype=out_dtype)
        scale_rows_per_cta = 2 if M_per_group % 2 == 0 else 1
        scale_k_unroll = 4
        scale_output_aligned = (
            M % scale_rows_per_cta == 0 and N % scale_threads_per_block == 0
        )
        scale_output = _compile_shared_scale_output_kernel(
            scale_threads_per_block,
            int(M_per_group),
            partial_dtype_name,
            _torch_dtype_name(out_dtype),
            int(scale_chunk_blocks),
            int(scale_rows_per_cta),
            int(scale_k_unroll),
            bool(scale_output_aligned),
        )
    if use_split_scale_partials:
        scale_partials = partials
        scale_accum_out = accum
        scale_out = out
        scale_a_s = a_s
        scale_b_s = b_s
    elif use_wrapped_scale_tensors:
        scale_partials = cached_scale_partials
        if scale_partials is None:
            scale_partials = _make_cutedsl_tensor(
                partials, partial_cutlass_dtype, leading_dim=2
            )
        scale_accum_out = (
            _make_cutedsl_tensor(
                accum, _cutlass_dtype(scale_accum_dtype_name), leading_dim=1
            )
            if accum is not None
            else None
        )
        scale_out = (
            _make_cutedsl_tensor(out, cutlass.BFloat16, leading_dim=1)
            if out is not None
            else None
        )
        scale_a_s = _make_cutedsl_tensor(a_s, cutlass.Float32, leading_dim=0)
        scale_b_s = _get_cached_b_scale_tensor(b_s, cutlass.Float32)
    else:
        scale_partials = partials
        scale_accum_out = accum
        scale_out = out
        scale_a_s = a_s
        scale_b_s = b_s
    for chunk_start in range(0, k_blocks, scale_chunk_blocks):
        chunk_blocks = min(scale_chunk_blocks, k_blocks - chunk_start)
        if use_kblock_batched_dense:
            storage_offset = chunk_start * block_size
            a_view = torch.as_strided(
                a,
                (M_per_group, block_size, E, chunk_blocks),
                (K, 1, M_per_group * K, block_size),
                storage_offset=storage_offset,
            )
            dense_key = (
                "expert_kblock_batched_persistent"
                if use_persistent_dense
                else "expert_kblock_batched",
                *key,
                chunk_blocks,
                partial_dtype_name,
            )
            compiled = (
                _HOPPER_DENSE_PERSISTENT_KBLOCK_BATCHED_GEMM_COMPILED.get(dense_key)
                if use_persistent_dense
                else _HOPPER_DENSE_KBLOCK_BATCHED_GEMM_COMPILED.get(dense_key)
            )
            b_view, mB = _get_cached_hopper_dense_b_kblock_batched_tensor(
                b,
                block_size,
                chunk_start,
                chunk_blocks,
                cutlass.Float8E4M3FN,
                compiled is None,
            )
            c_view = batched_c_view
            mC = batched_mC

            if compiled is None:
                dense_compile = (
                    _get_hopper_dense_persistent_kblock_batched_gemm
                    if use_persistent_dense
                    else _get_hopper_dense_kblock_batched_gemm
                )
                compiled = dense_compile(
                    a_view,
                    b_view,
                    c_view,
                    dense_key,
                    tile_shape_mn,
                    partial_cutlass_dtype,
                )
                if compiled is None:
                    return None

            mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
            compiled(mA, mB, mC, stream)
        else:
            for local_k in range(chunk_blocks):
                k_block = chunk_start + local_k
                storage_offset = k_block * block_size
                a_view = torch.as_strided(
                    a,
                    (M_per_group, block_size, E),
                    (K, 1, M_per_group * K),
                    storage_offset=storage_offset,
                )

                if compiled is None:
                    b_view = torch.as_strided(
                        b,
                        (N, block_size, E),
                        (K, 1, K * N),
                        storage_offset=storage_offset,
                    )
                    dense_key = (
                        "expert_kblock_persistent"
                        if use_persistent_dense
                        else "expert_kblock",
                        *key,
                        partial_dtype_name,
                    )
                    dense_compile = (
                        _get_hopper_dense_persistent_kblock_gemm
                        if use_persistent_dense
                        else _get_hopper_dense_kblock_gemm
                    )
                    compiled = dense_compile(
                        a_view,
                        b_view,
                        c_views[local_k],
                        dense_key,
                        tile_shape_mn,
                        partial_cutlass_dtype,
                    )
                    if compiled is None:
                        return None

                mA = _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN)
                compiled(mA, mBs[k_block], mCs[local_k], stream)
        if use_shared_scale_accum:
            if chunk_start + chunk_blocks == k_blocks and scale_output is not None:
                scale_output(
                    scale_partials,
                    scale_a_s,
                    scale_b_s,
                    scale_out,
                    int(M),
                    int(N),
                    stream,
                )
            else:
                if (
                    chunk_start + chunk_blocks == k_blocks
                    and scale_accum_output is not None
                ):
                    if use_split_scale_partials:
                        if use_split16_direct_scale_output:
                            scale_accum_output(
                                scale_partials[0],
                                scale_partials[1],
                                scale_partials[2],
                                scale_partials[3],
                                scale_partials[4],
                                scale_partials[5],
                                scale_partials[6],
                                scale_partials[7],
                                scale_partials[8],
                                scale_partials[9],
                                scale_partials[10],
                                scale_partials[11],
                                scale_partials[12],
                                scale_partials[13],
                                scale_partials[14],
                                scale_partials[15],
                                scale_a_s,
                                scale_b_s,
                                scale_out,
                                int(M),
                                int(N),
                                stream,
                            )
                        elif scale_chunk_blocks == 8:
                            scale_accum_output(
                                scale_partials[0],
                                scale_partials[1],
                                scale_partials[2],
                                scale_partials[3],
                                scale_partials[4],
                                scale_partials[5],
                                scale_partials[6],
                                scale_partials[7],
                                scale_accum_out,
                                scale_a_s,
                                scale_b_s,
                                scale_out,
                                int(M),
                                int(N),
                                int(chunk_start),
                                stream,
                            )
                        else:
                            scale_accum_output(
                                scale_partials[0],
                                scale_partials[1],
                                scale_partials[2],
                                scale_partials[3],
                                scale_accum_out,
                                scale_a_s,
                                scale_b_s,
                                scale_out,
                                int(M),
                                int(N),
                                int(chunk_start),
                                stream,
                            )
                    else:
                        scale_accum_output(
                            scale_partials,
                            scale_accum_out,
                            scale_a_s,
                            scale_b_s,
                            scale_out,
                            int(M),
                            int(N),
                            int(chunk_start),
                            int(chunk_blocks),
                            stream,
                        )
                else:
                    if use_split_scale_partials:
                        scale_accum_kernel = (
                            scale_accum_first if chunk_start == 0 else scale_accum
                        )
                        if chunk_start != 0 and scale_accum_static is not None:
                            scale_accum_kernel = scale_accum_static.get(
                                chunk_start, scale_accum
                            )
                        if scale_chunk_blocks == 8:
                            scale_accum_kernel(
                                scale_partials[0],
                                scale_partials[1],
                                scale_partials[2],
                                scale_partials[3],
                                scale_partials[4],
                                scale_partials[5],
                                scale_partials[6],
                                scale_partials[7],
                                scale_accum_out,
                                scale_a_s,
                                scale_b_s,
                                int(M),
                                int(N),
                                int(chunk_start),
                                stream,
                            )
                        else:
                            scale_accum_kernel(
                                scale_partials[0],
                                scale_partials[1],
                                scale_partials[2],
                                scale_partials[3],
                                scale_accum_out,
                                scale_a_s,
                                scale_b_s,
                                int(M),
                                int(N),
                                int(chunk_start),
                                stream,
                            )
                    else:
                        scale_accum_kernel = (
                            scale_accum_first if chunk_start == 0 else scale_accum
                        )
                        if chunk_start != 0 and scale_accum_static is not None:
                            scale_accum_kernel = scale_accum_static.get(
                                chunk_start, scale_accum
                            )
                        scale_accum_kernel(
                            scale_partials,
                            scale_accum_out,
                            scale_a_s,
                            scale_b_s,
                            int(M),
                            int(N),
                            int(chunk_start),
                            int(chunk_blocks),
                            stream,
                        )
        else:
            scale_accum(
                scale_partials,
                scale_accum_out,
                scale_a_s,
                scale_b_s,
                int(M),
                int(N),
                int(M_per_group),
                int(chunk_start),
                int(chunk_blocks),
                stream,
            )

    if out is not None:
        return out
    if out_dtype == torch.float32:
        return accum
    return accum.to(out_dtype)


@functools.cache
def _compile_wgrad_bf16_partial_scale_output(threads_per_block: int):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    class WgradBf16PartialScaleOutput:
        @cute.kernel
        def kernel(
            self,
            partials: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            E: cutlass.Int64,
            N: cutlass.Int64,
            K: cutlass.Int64,
            K_BLOCKS: cutlass.Int32,
            THREADS_PER_BLOCK: cutlass.Constexpr[int],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            cta_idx = cutlass.Int64(cta_idx)
            k_tiles = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            tiles_per_expert = N * k_tiles
            expert = cta_idx // tiles_per_expert
            expert_tile = cta_idx - expert * tiles_per_expert
            n = expert_tile // k_tiles
            k_tile = expert_tile - n * k_tiles
            k = cutlass.Int64(k_tile * THREADS_PER_BLOCK + tidx)

            if expert < E and n < N and k < K:
                acc = cutlass.Float32(0.0)
                for k_block in cutlass.range(0, K_BLOCKS, 1, unroll=1):
                    scale_block = cutlass.Int64(expert * K_BLOCKS + k_block)
                    acc += (
                        cutlass.Float32(partials[expert, k_block, n, k])
                        * cutlass.Float32(a_s[n, scale_block])
                        * cutlass.Float32(b_s[scale_block, k])
                    )
                out[expert, n, k] = cutlass.BFloat16(acc)

        @cute.jit
        def __call__(
            self,
            partials: cute.Tensor,
            a_s: cute.Tensor,
            b_s: cute.Tensor,
            out: cute.Tensor,
            E: cutlass.Int64,
            N: cutlass.Int64,
            K: cutlass.Int64,
            K_BLOCKS: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                partials,
                a_s,
                b_s,
                out,
                E,
                N,
                K,
                K_BLOCKS,
                THREADS_PER_BLOCK=threads_per_block,
            ).launch(
                grid=(E * N * cute.ceil_div(K, threads_per_block), 1, 1),
                block=(threads_per_block, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )

    e = cute.sym_int()
    k_blocks = cute.sym_int()
    n = cute.sym_int()
    k = cute.sym_int()
    scale_blocks = cute.sym_int()
    partials_stride0 = cute.sym_int()
    partials_stride1 = cute.sym_int()
    partials_stride2 = cute.sym_int()
    partials_stride3 = cute.sym_int()
    a_s_stride0 = cute.sym_int()
    a_s_stride1 = cute.sym_int()
    b_s_stride0 = cute.sym_int()
    b_s_stride1 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()
    out_stride2 = cute.sym_int()
    fake_partials = make_fake_tensor(
        cutlass.BFloat16,
        (e, k_blocks, n, k),
        stride=(
            partials_stride0,
            partials_stride1,
            partials_stride2,
            partials_stride3,
        ),
    )
    fake_a_s = make_fake_tensor(
        cutlass.Float32,
        (n, scale_blocks),
        stride=(a_s_stride0, a_s_stride1),
    )
    fake_b_s = make_fake_tensor(
        cutlass.Float32,
        (scale_blocks, k),
        stride=(b_s_stride0, b_s_stride1),
    )
    fake_out = make_fake_tensor(
        cutlass.BFloat16,
        (e, n, k),
        stride=(out_stride0, out_stride1, out_stride2),
    )
    fake_stream = make_fake_stream()
    return cute.compile(
        WgradBf16PartialScaleOutput(),
        partials=fake_partials,
        a_s=fake_a_s,
        b_s=fake_b_s,
        out=fake_out,
        E=0,
        N=0,
        K=0,
        K_BLOCKS=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def _cutedsl_fused_equal_group_wgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor | None:
    if out_dtype != torch.bfloat16 or not _cutedsl_runtime_available():
        return None

    N, M = a.shape
    _, K = b.shape
    E = offs.numel()
    M_per_group = _equal_group_size_from_offsets(offs, M)
    if M_per_group is None or M_per_group % block_size != 0:
        return None
    scale_k_blocks = M_per_group // block_size
    if E < 4 or N % 128 != 0:
        return None

    tile_shape_mn = (128, 128)
    if K % tile_shape_mn[1] != 0:
        return None

    a_view = torch.as_strided(
        a,
        (N, M_per_group, E),
        (M, 1, M_per_group),
    )
    b_view = torch.as_strided(
        b,
        (K, M_per_group, E),
        (M, 1, M_per_group),
    )
    out = torch.empty((E, N, K), dtype=out_dtype, device=a.device)
    c_view = torch.as_strided(
        out,
        (N, K, E),
        (K, 1, N * K),
    )
    a_s_grouped = a_s.t()
    b_s_grouped = b_s
    fused_key = (
        "wgrad_blockwise_scaled_persistent",
        M_per_group,
        N,
        K,
        E,
        block_size,
        scale_k_blocks,
        tile_shape_mn,
    )
    compiled_info = _get_hopper_blockwise_scaled_persistent_gemm(
        a_view,
        b_view,
        a_s_grouped,
        b_s_grouped,
        c_view,
        fused_key,
        tile_shape_mn,
        scale_k_blocks,
        wgrad=True,
    )
    if compiled_info is None:
        return None

    import cutlass

    compiled, _ = compiled_info
    stream = _get_cuda_stream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN),
        _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN),
        _make_cutedsl_tensor(a_s_grouped, cutlass.Float32, leading_dim=1),
        _make_cutedsl_tensor(b_s_grouped, cutlass.Float32, leading_dim=1),
        _make_hopper_dense_tensor(c_view, cutlass.BFloat16),
        stream,
    )
    return out


def _cutedsl_equal_group_wgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor | None:
    fused_out = _cutedsl_fused_equal_group_wgrad(
        a,
        b,
        a_s,
        b_s,
        offs,
        out_dtype,
        block_size,
    )
    if fused_out is not None:
        return fused_out
    if out_dtype != torch.bfloat16 or not _hopper_dense_gemm_available():
        return None

    N, M = a.shape
    _, K = b.shape
    E = offs.numel()
    M_per_group = _equal_group_size_from_offsets(offs, M)
    if M_per_group is None or M_per_group % block_size != 0:
        return None
    k_blocks = M_per_group // block_size
    workspace_bytes = E * k_blocks * N * K * torch.bfloat16.itemsize
    if workspace_bytes > _SCALE_ACCUM_MAX_PARTIAL_BYTES:
        return None

    stream_id = int(torch.cuda.current_stream().cuda_stream)
    workspace_key = (a.device, stream_id, E, k_blocks, N, K)
    partials = _WGRAD_BF16_PARTIAL_WORKSPACE_CACHE.get(workspace_key)
    if partials is None:
        partials = torch.empty(
            (E, k_blocks, N, K),
            dtype=torch.bfloat16,
            device=a.device,
        )
        if (
            len(_WGRAD_BF16_PARTIAL_WORKSPACE_CACHE)
            >= _WGRAD_BF16_PARTIAL_WORKSPACE_CACHE_MAX_ENTRIES
        ):
            _WGRAD_BF16_PARTIAL_WORKSPACE_CACHE.pop(
                next(iter(_WGRAD_BF16_PARTIAL_WORKSPACE_CACHE)),
                None,
            )
        _WGRAD_BF16_PARTIAL_WORKSPACE_CACHE[workspace_key] = partials

    a_view = torch.as_strided(
        a,
        (N, block_size, E, k_blocks),
        (M, 1, M_per_group, block_size),
    )
    b_view = torch.as_strided(
        b,
        (K, block_size, E, k_blocks),
        (M, 1, M_per_group, block_size),
    )
    c_view = torch.as_strided(
        partials,
        (N, K, E, k_blocks),
        (K, 1, k_blocks * N * K, N * K),
    )

    import cutlass

    tile_shape_mn = _hopper_dense_tile_shape_mn(K)
    gemm_key = (
        "wgrad_kblock_batched",
        E,
        M_per_group,
        N,
        K,
        tile_shape_mn,
    )
    compiled = _get_hopper_dense_kblock_batched_gemm(
        a_view,
        b_view,
        c_view,
        gemm_key,
        tile_shape_mn,
        cutlass.BFloat16,
    )
    if compiled is None:
        return None

    stream = _get_cuda_stream(stream_id)
    compiled(
        _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN),
        _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN),
        _make_hopper_dense_tensor(c_view, cutlass.BFloat16),
        stream,
    )
    out = torch.empty((E, N, K), dtype=out_dtype, device=a.device)
    scale_output = _compile_wgrad_bf16_partial_scale_output(256)
    scale_output(
        partials,
        a_s,
        b_s,
        out,
        int(E),
        int(N),
        int(K),
        int(k_blocks),
        stream,
    )
    return out


@torch.library.custom_op(
    "torchao::cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d",
    mutates_args=(),
)
def cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    if not _cutedsl_runtime_available():
        missing = ", ".join(_missing_cutedsl_runtime_packages())
        raise RuntimeError(f"CuTeDSL runtime packages are not available: {missing}")
    assert a.ndim == 2, "a must be 2D"
    assert b.ndim == 2, "b must be 2D"
    assert a.dtype == torch.float8_e4m3fn, "a must be torch.float8_e4m3fn"
    assert b.dtype == torch.float8_e4m3fn, "b must be torch.float8_e4m3fn"
    assert a_s.dtype == torch.float32 and b_s.dtype == torch.float32
    assert offs.dtype == torch.int32, "offs must be int32"
    assert block_size == 128, "Only block_size=128 is supported"
    assert out_dtype in (torch.bfloat16, torch.float32), (
        "out_dtype must be bfloat16 or float32"
    )
    assert a.stride(1) == 1, "a must be row-major"
    assert b.stride(0) == 1, "b must be column-major"
    assert a_s.stride(0) == 1, "a_s must be column-major"
    assert b_s.stride(1) == 1, "b_s must be row-major"

    N, M = a.shape
    b_m, K = b.shape
    E = offs.numel()
    assert M == b_m, f"shape {a.shape} and {b.shape} are not compatible"
    assert M % block_size == 0, "the grouped reduction extent must be block-aligned"
    assert a_s.shape == (N, M // block_size), (
        "a_s must have shape (a.shape[0], a.shape[1] // block_size)"
    )
    assert b_s.shape == (M // block_size, K), (
        "b_s must have shape (b.shape[0] // block_size, b.shape[1])"
    )
    assert _wgrad_offsets_are_valid(offs, M, block_size), (
        "offs must be nondecreasing, block-aligned, nonempty, and within the inputs"
    )

    fast_out = _cutedsl_equal_group_wgrad(
        a,
        b,
        a_s,
        b_s,
        offs,
        out_dtype,
        block_size,
    )
    if fast_out is not None:
        return fast_out

    out = a.new_empty((E, N, K), dtype=out_dtype)
    compiled = _compile_fp8_blockwise_grouped_gemm_2d_2d(
        _torch_dtype_name(a.dtype),
        _torch_dtype_name(b.dtype),
        _torch_dtype_name(out_dtype),
        block_size,
        128,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        a,
        b,
        a_s,
        b_s,
        offs,
        out,
        int(E),
        int(N),
        int(K),
        stream,
    )
    return out


@cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    return a.new_empty((offs.numel(), a.shape[0], b.shape[-1]), dtype=out_dtype)


@torch.library.custom_op(
    "torchao::cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d",
    mutates_args=(),
)
def cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    if not _cutedsl_runtime_available():
        missing = ", ".join(_missing_cutedsl_runtime_packages())
        raise RuntimeError(f"CuTeDSL runtime packages are not available: {missing}")
    assert a.ndim == 2, "a must be 2D"
    assert b.ndim == 3, "b must be 3D"
    assert a.dtype == torch.float8_e4m3fn, "a must be torch.float8_e4m3fn"
    assert b.dtype == torch.float8_e4m3fn, "b must be torch.float8_e4m3fn"
    assert a_s.dtype == torch.float32 and b_s.dtype == torch.float32
    assert offs.dtype == torch.int32, "offs must be int32"
    assert block_size == 128, "Only block_size=128 is supported"
    assert out_dtype in (torch.bfloat16, torch.float32), (
        "out_dtype must be bfloat16 or float32"
    )

    M, K = a.shape
    E, b_k, N = b.shape
    assert K == b_k, f"shape {a.shape} and {b.shape} are not compatible"
    assert offs.numel() == E, "offs must have one end offset per expert"
    assert a_s.shape == (M, ceil_div(K, block_size)), (
        "a_s must have shape (M, K // block_size)"
    )
    assert b_s.shape[0] == E and b_s.shape[2] == ceil_div(N, block_size), (
        "b_s must have shape (E, padded_K_blocks, N // block_size)"
    )

    if K >= 1024:
        splitk_out = _cutedsl_splitk_equal_group_scaled_gemm(
            a,
            b,
            a_s,
            b_s,
            offs,
            out_dtype,
            block_size,
        )
        if splitk_out is not None:
            return splitk_out

    grouped_layout = _m_grouped_layout_from_offsets(offs, M)
    out = a.new_empty((M, N), dtype=out_dtype)
    compiled = _compile_fp8_blockwise_grouped_gemm_2d_3d(
        _torch_dtype_name(a.dtype),
        _torch_dtype_name(b.dtype),
        _torch_dtype_name(out_dtype),
        block_size,
        128,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        a,
        b,
        a_s,
        b_s,
        grouped_layout,
        out,
        int(M),
        int(N),
        int(K),
        stream,
    )
    return out


@cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[-1]), dtype=out_dtype)


def maybe_cutedsl_fp8_blockwise_scaled_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[bool, torch.Tensor | None]:
    if _is_cutedsl_2d_2d_supported(
        a,
        b,
        a_s,
        scale_recipe_a,
        b_s,
        scale_recipe_b,
        offs,
        out_dtype,
        block_size,
    ):
        return (
            True,
            cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d(
                a,
                b,
                a_s,
                b_s,
                offs,
                out_dtype,
                block_size,
            ),
        )
    if not _is_cutedsl_2d_3d_supported(
        a,
        b,
        a_s,
        scale_recipe_a,
        b_s,
        scale_recipe_b,
        offs,
        out_dtype,
        block_size,
    ):
        return False, None
    return (
        True,
        cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d(
            a,
            b,
            a_s,
            b_s,
            offs,
            out_dtype,
            block_size,
        ),
    )
