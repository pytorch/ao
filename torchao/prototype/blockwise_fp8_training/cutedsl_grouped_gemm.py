# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import weakref

import torch

from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _cutedsl_runtime_available,
    _missing_cutedsl_runtime_packages,
)

_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM_ENV = (
    "TORCHAO_ENABLE_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM"
)
_HOPPER_BLOCKWISE_SCALED_PERSISTENT_GEMM_COMPILED = {}
_CUDA_STREAM_CACHE = {}
_EQUAL_GROUP_OFFSETS_CACHE = {}
_HOPPER_DENSE_TILE_SHAPE_MN = (128, 128)
_HOPPER_DENSE_WIDE_N_TILE_SHAPE_MN = (128, 256)
_HOPPER_DENSE_FWD_TILE_SHAPE_MN = (128, 192)


def _cutedsl_fp8_blockwise_grouped_mm_enabled() -> bool:
    return os.environ.get(_CUTEDSL_FP8_BLOCKWISE_GROUPED_MM_ENV, "0") == "1"


@functools.cache
def _load_cutedsl_hopper_gemm_module():
    if not _cutedsl_runtime_available():
        return None

    from torchao.prototype.blockwise_fp8_training import _cutedsl_hopper_gemm

    return _cutedsl_hopper_gemm


def _load_hopper_dense_gemm_persistent_module():
    return _load_cutedsl_hopper_gemm_module()


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


def _fused_2d_3d_tile_shape(K: int, N: int) -> tuple[int, int] | None:
    reduction_blocks = K // 128
    if K % 128 != 0 or reduction_blocks not in (16, 56):
        return None
    if reduction_blocks == 56 and N >= _HOPPER_DENSE_FWD_TILE_SHAPE_MN[1]:
        return _HOPPER_DENSE_FWD_TILE_SHAPE_MN
    return _hopper_dense_tile_shape_mn(N)


def _fused_2d_3d_geometry_supported(
    M_per_group: int,
    K: int,
    N: int,
) -> bool:
    tile_shape_mn = _fused_2d_3d_tile_shape(K, N)
    return bool(
        tile_shape_mn is not None
        and M_per_group % tile_shape_mn[0] == 0
        and (
            tile_shape_mn == _HOPPER_DENSE_FWD_TILE_SHAPE_MN
            or N % tile_shape_mn[1] == 0
        )
    )


def _common_fused_support(
    tensors: tuple[torch.Tensor, ...],
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> bool:
    if not (
        _cutedsl_fp8_blockwise_grouped_mm_enabled()
        and _cutedsl_runtime_available()
        and torch.cuda.is_available()
        and not torch.version.hip
        and tensors
        and all(t.is_cuda for t in tensors)
        and offs.is_cuda
        and all(t.device == tensors[0].device for t in tensors[1:])
        and offs.device == tensors[0].device
        and offs.ndim == 1
        and offs.dtype == torch.int32
        and offs.numel() >= 4
        and out_dtype == torch.bfloat16
        and block_size == 128
    ):
        return False
    major, _ = torch.cuda.get_device_capability(tensors[0].device)
    return major >= 9


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
    if not (
        _common_fused_support((a, b, a_s, b_s), offs, out_dtype, block_size)
        and a.ndim == 2
        and b.ndim == 3
        and a_s.ndim == 2
        and b_s.ndim == 3
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == torch.float8_e4m3fn
        and a_s.dtype == torch.float32
        and b_s.dtype == torch.float32
        and scale_recipe_a == 4
        and scale_recipe_b == 5
    ):
        return False

    M, K = a.shape
    E, b_k, N = b.shape
    M_per_group = _equal_group_size_from_offsets(offs, M)
    k_blocks = K // block_size
    n_blocks = N // block_size
    return bool(
        E == offs.numel()
        and K == b_k
        and M_per_group is not None
        and N % block_size == 0
        and _fused_2d_3d_geometry_supported(M_per_group, K, N)
        and a.stride() == (K, 1)
        and b.stride() == (K * N, 1, K)
        and a_s.shape == (M, k_blocks)
        and a_s.stride() == (1, M)
        and b_s.shape == (E, k_blocks, n_blocks)
        and b_s.stride() == (k_blocks * n_blocks, 1, k_blocks)
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
        _common_fused_support((a, b, a_s, b_s), offs, out_dtype, block_size)
        and a.ndim == 2
        and b.ndim == 2
        and a_s.ndim == 2
        and b_s.ndim == 2
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == torch.float8_e4m3fn
        and a_s.dtype == torch.float32
        and b_s.dtype == torch.float32
        and scale_recipe_a == 4
        and scale_recipe_b == 4
    ):
        return False

    N, M = a.shape
    b_m, K = b.shape
    M_per_group = _equal_group_size_from_offsets(offs, M)
    m_blocks = M // block_size
    return bool(
        M == b_m
        and M_per_group is not None
        and M_per_group % block_size == 0
        and N % block_size == 0
        and K % block_size == 0
        and a.stride() == (M, 1)
        and b.stride() == (1, M)
        and a_s.shape == (N, m_blocks)
        and a_s.stride() == (1, N)
        and b_s.shape == (m_blocks, K)
        and b_s.stride() == (K, 1)
    )


def _can_use_cutedsl_fp8_blockwise_grouped_mm_training(
    a: torch.Tensor,
    b_t: torch.Tensor,
    group_end_offsets: torch.Tensor,
    original_group_end_offsets: torch.Tensor,
    out_dtype: torch.dtype,
    float8_dtype: torch.dtype,
    block_size: int,
    num_rows: int,
) -> bool:
    if not (
        _common_fused_support((a, b_t), group_end_offsets, out_dtype, block_size)
        and original_group_end_offsets.is_cuda
        and original_group_end_offsets.device == a.device
        and original_group_end_offsets.dtype == torch.int32
        and original_group_end_offsets.ndim == 1
        and original_group_end_offsets.numel() == group_end_offsets.numel()
        and a.ndim == 2
        and b_t.ndim == 3
        and a.dtype in (torch.bfloat16, torch.float32)
        and b_t.dtype in (torch.bfloat16, torch.float32)
        and float8_dtype == torch.float8_e4m3fn
        and a.stride(-1) == 1
        and b_t.stride(-2) == 1
        and a.shape[-1] == b_t.shape[-2]
        and b_t.shape[0] == group_end_offsets.numel()
        and num_rows == a.shape[0]
    ):
        return False

    E, K, N = b_t.shape
    M_per_group = _equal_group_size_from_offsets(group_end_offsets, num_rows)
    original_M_per_group = _equal_group_size_from_offsets(
        original_group_end_offsets, a.shape[0]
    )
    return bool(
        E >= 4
        and M_per_group is not None
        and M_per_group == original_M_per_group
        and M_per_group % block_size == 0
        and _fused_2d_3d_geometry_supported(M_per_group, K, N)
        and _fused_2d_3d_geometry_supported(M_per_group, N, K)
        and N % block_size == 0
        and K % block_size == 0
    )


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
                    tile_n_scale_offset = cutlass.Int64(0)
                    if cutlass.const_expr(self.tile_shape_mnk[1] == 192):
                        tile_n_start = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                        n_block_base = tile_n_start // BLOCK_SIZE
                        tile_n_scale_offset = tile_n_start - n_block_base * BLOCK_SIZE
                    else:
                        n_block_base = tile_coord_mnl[1] * scale_blocks_n
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


def _cutedsl_fused_equal_group_scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    M, K = a.shape
    E, _, N = b.shape
    M_per_group = M // E
    k_blocks = K // block_size
    tile_shape_mn = _fused_2d_3d_tile_shape(K, N)
    assert tile_shape_mn is not None

    import cutlass

    stream = _get_cuda_stream(int(torch.cuda.current_stream().cuda_stream))
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
        tile_shape_mn,
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
        tile_shape_mn,
        k_blocks,
    )
    if compiled_info is None:
        raise RuntimeError("CuTeDSL Hopper persistent grouped GEMM failed to compile")

    compiled, _ = compiled_info
    compiled(
        _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN),
        _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN),
        _make_cutedsl_tensor(a_s_view, cutlass.Float32, leading_dim=1),
        _make_cutedsl_tensor(b_s, cutlass.Float32, leading_dim=1),
        _make_hopper_dense_tensor(c_view, cutlass.BFloat16),
        stream,
    )
    return out


def _cutedsl_fused_equal_group_wgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    N, M = a.shape
    _, K = b.shape
    E = offs.numel()
    M_per_group = M // E
    scale_k_blocks = M_per_group // block_size
    tile_shape_mn = (128, 128)

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
        b_s,
        c_view,
        fused_key,
        tile_shape_mn,
        scale_k_blocks,
        wgrad=True,
    )
    if compiled_info is None:
        raise RuntimeError("CuTeDSL Hopper persistent wgrad GEMM failed to compile")

    import cutlass

    compiled, _ = compiled_info
    stream = _get_cuda_stream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        _make_hopper_dense_tensor(a_view, cutlass.Float8E4M3FN),
        _make_hopper_dense_tensor(b_view, cutlass.Float8E4M3FN),
        _make_cutedsl_tensor(a_s_grouped, cutlass.Float32, leading_dim=1),
        _make_cutedsl_tensor(b_s, cutlass.Float32, leading_dim=1),
        _make_hopper_dense_tensor(c_view, cutlass.BFloat16),
        stream,
    )
    return out


def _unsupported_cutedsl_message(operation: str) -> str:
    if not _cutedsl_runtime_available():
        missing = ", ".join(_missing_cutedsl_runtime_packages())
        return f"CuTeDSL runtime packages are not available: {missing}"
    return (
        f"CuTeDSL {operation} only supports opt-in CUDA Hopper+ fused equal-group "
        "FP8 blockwise GEMM with bfloat16 output and block_size=128"
    )


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
    if not _is_cutedsl_2d_2d_supported(
        a,
        b,
        a_s,
        4,
        b_s,
        4,
        offs,
        out_dtype,
        block_size,
    ):
        raise NotImplementedError(_unsupported_cutedsl_message("2D x 2D grouped GEMM"))
    return _cutedsl_fused_equal_group_wgrad(
        a,
        b,
        a_s,
        b_s,
        offs,
        out_dtype,
        block_size,
    )


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
    assert a.ndim == b.ndim == a_s.ndim == b_s.ndim == 2
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a_s.dtype == b_s.dtype == torch.float32
    assert offs.ndim == 1 and offs.dtype == torch.int32
    assert a.shape[1] == b.shape[0]
    assert out_dtype == torch.bfloat16 and block_size == 128
    return a.new_empty((offs.numel(), a.shape[0], b.shape[1]), dtype=out_dtype)


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
    if not _is_cutedsl_2d_3d_supported(
        a,
        b,
        a_s,
        4,
        b_s,
        5,
        offs,
        out_dtype,
        block_size,
    ):
        raise NotImplementedError(_unsupported_cutedsl_message("2D x 3D grouped GEMM"))
    return _cutedsl_fused_equal_group_scaled_gemm(
        a,
        b,
        a_s,
        b_s,
        offs,
        out_dtype,
        block_size,
    )


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
    assert a.ndim == a_s.ndim == 2 and b.ndim == b_s.ndim == 3
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a_s.dtype == b_s.dtype == torch.float32
    assert offs.ndim == 1 and offs.dtype == torch.int32
    assert a.shape[1] == b.shape[1] and offs.numel() == b.shape[0]
    assert out_dtype == torch.bfloat16 and block_size == 128
    return a.new_empty((a.shape[0], b.shape[2]), dtype=out_dtype)


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
) -> tuple[bool, torch.Tensor | None]:
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
        return True, cutedsl_fp8_blockwise_scaled_grouped_mm_2d_2d(
            a,
            b,
            a_s,
            b_s,
            offs,
            out_dtype,
            block_size,
        )
    if _is_cutedsl_2d_3d_supported(
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
        return True, cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d(
            a,
            b,
            a_s,
            b_s,
            offs,
            out_dtype,
            block_size,
        )
    return False, None
