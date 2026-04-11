# pyrefly: ignore-errors
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# pyrefly: ignore-errors

import os
from inspect import isclass
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, extract_mlir_values, new_from_mlir_values
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

"""
This example provides an experimental implementation of the SM100 grouped blockscaled GEMM kernel, please note that the APIs and implementation details related to this kernel may change in future releases.

A grouped blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of grouped blockscaled GEMM using a TMA plus Blackwell SM100 TensorCore
warp-specialized persistent kernel.
The grouped GEMM workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
in global memory are passed to the kernel in an array (also held in global memory). Similarly, problem shapes and
strides are also stored in arrays in GMEM.

This differs from "Batched Array" GEMM since the size of each GEMM problem in the grouped GEMM concept may be distinct.

To run this example:

.. code-block:: bash

    python examples/blackwell/grouped_blockscaled_gemm.py                                     \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16                       \
      --c_dtype Float16                                                                       \
      --mma_tiler_mn 128,128 --cluster_shape_mn 1,1                                           \
      --problem_sizes_mnkl "(8192,1280,32,1),(32,384,1536,1),(640,1280,32,1),(640,160,32,1)"  \
      --num_groups 4

The above example command makes 4 groups of different m, n, k sizes. The Blackwell tcgen05 MMA tile shape
is specified as (128, 64) and the cluster shape is (1,1). The input, mma accumulator and output data type
are set as fp16, fp32 and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/grouped_blockscaled_gemm.py                                 \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16                       \
      --c_dtype Float16                                                                       \
      --mma_tiler_mn 128,128 --cluster_shape_mn 1,1                                           \
      --problem_sizes_mnkl "(8192,1280,32,1),(32,384,1536,1),(640,1280,32,1),(640,160,32,1)"  \
      --num_groups 4
      --warmup_iterations 1 --iterations 10 --skip_ref_check

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100GroupedBlockScaledGemmKernel class documentation
* A/B tensors must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The l mode(aka, batch size) for each group must be 1.
* The majorness for A, B and C must be the same across all groups.
* The contiguous dimension of A/B/C tensors in each group must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


class ClcDynamicPersistentGroupTileScheduler:
    """Wrapper around ClcDynamicPersistentTileScheduler for grouped GEMM.

    The CLC grid uses an x-major layout: tile count is encoded in grid x
    (limit 2^31-1) instead of grid z (limit 65535).  Grid shape is
    ``(est_total_clusters * cluster_mn[0], cluster_mn[1], 1)``.

    Both ``initial_work_tile_info`` and ``get_current_work`` produce
    tile coordinates ``(cta_pos_x + c*Cm, cta_pos_y, 0)`` in this
    layout.  ``_remap_x_to_z`` converts to the z-major format
    ``(cta_pos_x, cta_pos_y, c)`` expected by ``delinearize_z``.
    """

    def __init__(
        self,
        scheduler,
        helper_tile_sched_params,
        cluster_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        group_count: int,
        problem_shape_mnkl: cute.Tensor,
        total_num_clusters: cute.Tensor,
        cluster_shape_m_fdd: cute.FastDivmodDivisor,
    ):
        self.scheduler = scheduler
        self.helper_tile_sched_params = helper_tile_sched_params
        self.cluster_tile_shape_mnk = cluster_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn
        self.group_count = group_count
        self.problem_shape_mnkl = problem_shape_mnkl
        self.total_num_clusters = total_num_clusters
        self.cluster_shape_m_fdd = cluster_shape_m_fdd

    _mlir_fields = (
        "scheduler",
        "helper_tile_sched_params",
        "problem_shape_mnkl",
        "total_num_clusters",
        "cluster_shape_m_fdd",
    )

    def __extract_mlir_values__(self):
        values: list = []
        for name in self._mlir_fields:
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def __new_from_mlir_values__(self, values):
        rebuilt: dict = {}
        offset = 0
        for name in self._mlir_fields:
            old = getattr(self, name)
            n = len(extract_mlir_values(old))
            rebuilt[name] = new_from_mlir_values(old, values[offset : offset + n])
            offset += n
        return ClcDynamicPersistentGroupTileScheduler(
            rebuilt["scheduler"],
            rebuilt["helper_tile_sched_params"],
            self.cluster_tile_shape_mnk,
            self.cluster_shape_mn,
            self.group_count,
            rebuilt["problem_shape_mnkl"],
            rebuilt["total_num_clusters"],
            rebuilt["cluster_shape_m_fdd"],
        )

    def _make_group_helper(self):
        return utils.GroupedGemmTileSchedulerHelper(
            self.group_count,
            self.helper_tile_sched_params,
            self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
        )

    @dsl_user_op
    @cute.jit
    def _make_grouped_work_tile(self, work_tile, *, loc=None, ip=None):
        is_valid_tile = work_tile.is_valid_tile
        group_search_result = utils.GroupSearchResult(
            cutlass.Int32(-1),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(0),
        )
        if is_valid_tile:
            is_valid_tile = work_tile.tile_idx[2] < self.total_num_clusters[0]
            if is_valid_tile:
                group_search_result = self._make_group_helper().delinearize_z(
                    work_tile.tile_idx,
                    self.problem_shape_mnkl,
                )
        return utils.GroupedWorkTileInfo(
            work_tile.tile_idx,
            is_valid_tile,
            group_search_result,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params,
        block_idx: tuple[int, int, int],
        grid_dim: tuple[int, int, int],
        clc_response_ptr: cute.Pointer,
        helper_tile_sched_params,
        cluster_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        group_count: int,
        problem_shape_mnkl: cute.Tensor,
        total_num_clusters: cute.Tensor,
        *,
        loc=None,
        ip=None,
    ):
        scheduler = utils.ClcDynamicPersistentTileScheduler.create(
            params,
            block_idx,
            grid_dim,
            clc_response_ptr,
            loc=loc,
            ip=ip,
        )
        cluster_shape_m_fdd = cute.fast_divmod_create_divisor(
            cluster_shape_mn[0], loc=loc, ip=ip
        )
        return ClcDynamicPersistentGroupTileScheduler(
            scheduler,
            helper_tile_sched_params,
            cluster_tile_shape_mnk,
            cluster_shape_mn,
            group_count,
            problem_shape_mnkl,
            total_num_clusters,
            cluster_shape_m_fdd,
        )

    @dsl_user_op
    def _remap_x_to_z(self, work, *, loc=None, ip=None):
        """Convert x-major tile coordinates to z-major for delinearize_z.

        x-major tile_idx: (c * Cm + cta_pos_x, cta_pos_y, 0)
        z-major tile_idx: (cta_pos_x, cta_pos_y, c)
        """
        tile_x = work.tile_idx[0]
        linear_cluster_idx, cta_pos_m = divmod(tile_x, self.cluster_shape_m_fdd)
        cta_pos_n = work.tile_idx[1]
        return utils.WorkTileInfo(
            (cta_pos_m, cta_pos_n, linear_cluster_idx),
            work.is_valid_tile,
        )

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None):
        work = self.scheduler.get_current_work(loc=loc, ip=ip)
        return self._make_grouped_work_tile(self._remap_x_to_z(work))

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None):
        work = self.scheduler.initial_work_tile_info(loc=loc, ip=ip)
        return self._make_grouped_work_tile(self._remap_x_to_z(work))

    @dsl_user_op
    def advance_to_next_work(self, mbarrier_addr, *, loc=None, ip=None):
        self.scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)

    @property
    def num_tiles_executed(self):
        return self.scheduler.num_tiles_executed


class Sm100GroupedBlockScaledGemmKernel:
    """This example demonstrates an implementation of grouped blockscaled GEMM using a TMA plus Blackwell SM100 TensorCore
    warp-specialized persistent kernel.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensors must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
    """

    CLC_RESPONSE_BYTES = 16

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_clc: bool = False,
    ):
        """Initializes the configuration for a Blackwell grouped blockscaled GEMM kernel.

        Besides configurations for dense persistent blockscaled GEMM, there is an extra config specific to grouped blockscaled GEMM:

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        """
        self.use_clc = use_clc
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = utils.TensorMapUpdateMode.SMEM

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # Barrier used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=64,
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    # Set up configurations that dependent on gemm inputs.
    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Checking reserved smem bytes size capacity for mbar, tensor memory management and tensormap updates utilization
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        # mma_inst_tile_k is hardcoded to 4 in make_smem_layout_sfa/sfb
        # so if changed has to be changed everywhere
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_c_stage=self.num_c_stage,
            num_clc_stage=1,
        )
        # CLC currently uses a single stage, so reserve one 16-byte response slot.
        mbar_smem_bytes += Sm100GroupedBlockScaledGemmKernel.CLC_RESPONSE_BYTES

        # Use utils.TensorMapUpdateMode.SMEM by default
        tensormap_smem_bytes = (
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            * Sm100GroupedBlockScaledGemmKernel.num_tensormaps
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + Sm100GroupedBlockScaledGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

    @cute.jit
    def __call__(
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        tensor_sfa: cute.Tensor,
        tensor_sfb: cute.Tensor,
        cute_offs: cute.Tensor,
        a_offs: cute.Tensor | None,
        b_offs: cute.Tensor | None,
        out_offset: cute.Tensor | None,
        total_num_clusters: cute.Tensor,
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_address_sfasfb: cute.Tensor,
        est_total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: int,
        input_strides_abc: tuple[
            tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
        ],
        tile_sf_to_dynamic_dim: cutlass.Constexpr[bool],
        dynamic_dims: tuple[int, int],
        group_count: cutlass.Constexpr[int],
        M: cutlass.Constexpr[int],
        N: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        addmm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        For grouped GEMM, tensor shapes, tensor strides, and tensor address are all provided
        by different tensors in global memory. The tensors carry data type and majorness information.

        :param tensor_a: Tensor A, used for data type and majorness information.
        :type tensor_a: cute.Tensor
        :param tensor_b: Tensor B, used for data type and majorness information.
        :type tensor_b: cute.Tensor
        :param tensor_c: Tensor C, used for data type and majorness information.
        :type tensor_c: cute.Tensor
        :param tensor_sfa: Tensor SFA, used for data type and majorness information.
        :type tensor_sfa: cute.Tensor
        :param tensor_sfb: Tensor SFB, used for data type and majorness information.
        :type tensor_sfb: cute.Tensor
        :param cute_offs: Tensor containing the offsets for each group.
        :type cute_offs: cute.Tensor
        :param total_num_clusters: Tensor containing the total number of clusters.
        :type total_num_clusters: cute.Tensor
        :param problem_shape_mnkl: Tensor containing the (M, N, K, L) shape for each group.
        :type problem_shape_mnkl: cute.Tensor
        :param strides_abc: Tensor containing the strides for A, B, and C for each group.
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing the base addresses for A, B, and C for each group.
        :type tensor_address_abc: cute.Tensor
        :param tensor_address_sfasfb: Tensor containing the base addresses for SFA and SFB for each group.
        :type tensor_address_sfasfb: cute.Tensor
        :param est_total_num_clusters: Estimated total number of clusters needed for all groups.
        :type est_total_num_clusters: cutlass.Constexpr[int]
        :param tensormap_cute_tensor: Tensor for storing tensormaps.
        :type tensormap_cute_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: int
        :param input_strides_abc: Tuple of input strides for A, B, and C matrices, and batch strides if applicable
        :type input_strides_abc: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
        :param tile_sf_to_dynamic_dim: Whether to tile scale factors to dynamic dimension.
        :type tile_sf_to_dynamic_dim: cutlass.Constexpr[bool]
        :param dynamic_dims: Tuple of dynamic dimensions (M size for 2d-3D, totak K for 2d-2d)
        :type dynamic_dims: tuple[int, int]
        :param group_count: The number of GEMM groups.
        :type group_count: cutlass.Constexpr[int]
        :param M: The M dimension, -1 if M is dynamic
        :type M: cutlass.Constexpr[int]
        :param N: The N dimension.
        :type N: cutlass.Constexpr[int]
        :param K: The K dimension, -1 if K is dynamic
        :type K: cutlass.Constexpr[int]
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """
        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.sf_dtype = tensor_sfa.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            tensor_a.shape, self.sf_vec_size
        )
        tensor_sfa = cute.make_tensor(tensor_sfa.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            tensor_b.shape, self.sf_vec_size
        )
        tensor_sfb = cute.make_tensor(tensor_sfb.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            tensor_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            tensor_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            tensor_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            tensor_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            (
                cpasync.CopyReduceBulkTensorTileS2GOp()
                if addmm
                else cpasync.CopyBulkTensorTileS2GOp()
            ),
            tensor_c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        est_tile_sched_params = self._compute_tile_sched(
            est_total_num_clusters, self.cluster_shape_mn, use_clc=self.use_clc
        )

        grid = self._compute_grid(
            est_tile_sched_params,
            max_active_clusters,
            use_clc=self.use_clc,
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps
            * Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            // 8
        )

        # Define shared storage for kernel
        clc_response_i64 = Sm100GroupedBlockScaledGemmKernel.CLC_RESPONSE_BYTES // 8

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, clc_response_i64],
                16,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        problem_sizes_gmnk = (group_count, M, N, K)
        self.prepare_kernel(  # type: ignore[attr-defined]
            input_ptr_a=tensor_a.iterator,
            input_ptr_b=tensor_b.iterator,
            input_ptr_c=tensor_c.iterator,
            input_ptr_sfa=tensor_sfa.iterator,
            input_ptr_sfb=tensor_sfb.iterator,
            offs=cute_offs,
            a_offs=a_offs,
            b_offs=b_offs,
            out_offset=out_offset,
            input_problem_sizes_gmnk=problem_sizes_gmnk,
            input_strides_abc=input_strides_abc,
            dynamic_dims=dynamic_dims,
            output_problem_sizes_mnkl=problem_shape_mnkl,
            output_strides_abc=strides_abc,
            output_ptrs_abc=tensor_address_abc,
            output_ptrs_sfasfb=tensor_address_sfasfb,
            output_total_num_clusters=total_num_clusters,
        ).launch(
            grid=(group_count, 1, 1),
            block=(32, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

        # Launch the kernel synchronously
        self.kernel(  # type: ignore[attr-defined]
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            total_num_clusters,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensor_address_sfasfb,
            tensormap_cute_tensor,
            tile_sf_to_dynamic_dim,
            dynamic_dims[0],
            dynamic_dims[1],
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        total_num_clusters: cute.Tensor,
        group_count: cutlass.Constexpr,
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        ptrs_sfasfb: cute.Tensor,
        tensormaps: cute.Tensor,
        tile_sf_to_dynamic_dim: cutlass.Constexpr[bool],
        sfa_k_stride: cutlass.Int32,
        sfb_k_stride: cutlass.Int32,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        tile_sched_params = self._compute_tile_sched(
            total_num_clusters[0], self.cluster_shape_mn, use_clc=self.use_clc
        )
        helper_sched_params = (
            self._compute_tile_sched(
                total_num_clusters[0],
                self.cluster_shape_mn,
                use_clc=False,
            )
            if self.use_clc
            else tile_sched_params
        )

        #
        # Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_smem_ptr
        tensormap_b_smem_ptr = (
            tensormap_a_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfa_smem_ptr = (
            tensormap_b_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfb_smem_ptr = (
            tensormap_sfa_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_c_smem_ptr = (
            tensormap_sfb_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        clc_pipeline = None
        clc_response_ptr = None
        if cutlass.const_expr(self.use_clc):
            cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
            clc_warp_ids = (self.tma_warp_id, self.mma_warp_id, *self.epilog_warp_id)
            clc_consumer_arv_count = (
                cluster_size * len(clc_warp_ids) * cute.arch.WARP_SIZE
            )
            clc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            clc_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                clc_consumer_arv_count,
            )
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                num_stages=1,
                producer_group=clc_producer_group,
                consumer_group=clc_consumer_group,
                tx_count=Sm100GroupedBlockScaledGemmKernel.CLC_RESPONSE_BYTES,
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            cute.arch.mbarrier_init_fence()

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA Load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA Load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Get tensormap buffer address
        #
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
        )

        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap,
        )
        tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 3, None)].iterator
        )
        tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 4, None)].iterator
        )

        #
        # Persistent tile scheduling loop
        #
        if cutlass.const_expr(self.use_clc):
            tile_sched = ClcDynamicPersistentGroupTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                grid_dim,
                clc_response_ptr,
                helper_sched_params,
                self.cluster_tile_shape_mnk,
                self.cluster_shape_mn,
                group_count,
                problem_sizes_mnkl,
                total_num_clusters,
            )
        else:
            tile_sched = utils.StaticPersistentGroupTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                grid_dim,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
                group_count,
                problem_sizes_mnkl,
            )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            tensormap_init_done = cutlass.Boolean(False)
            last_group_idx = cutlass.Int32(-1)

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            clc_producer_state = None
            clc_consumer_state = None
            if cutlass.const_expr(self.use_clc):
                clc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    1,
                )
                clc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    1,
                )

            while work_tile.is_valid_tile:  # type: ignore
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    real_tensor_a = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        0,  # 0 for tensor A
                    )
                    real_tensor_b = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        1,  # 1 for tensor B
                    )
                    shape_k = cutlass.Int32(-1)
                    if cutlass.const_expr(tile_sf_to_dynamic_dim):
                        shape_k = sfa_k_stride
                    else:
                        shape_k = grouped_gemm_cta_tile_info.problem_shape_k
                    real_tensor_sfa = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            shape_k,
                        ),
                        ptrs_sfasfb,
                        0,  # 0 for tensor SFA
                    )
                    if cutlass.const_expr(tile_sf_to_dynamic_dim):
                        shape_k = sfb_k_stride
                    else:
                        shape_k = grouped_gemm_cta_tile_info.problem_shape_k

                    real_tensor_sfb = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            shape_k,
                        ),
                        ptrs_sfasfb,
                        1,  # 1 for tensor SFB
                    )
                    if tensormap_init_done == False:
                        # wait tensormap initialization complete
                        self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_init_done = True

                    tensormap_manager.update_tensormap(
                        (
                            real_tensor_a,
                            real_tensor_b,
                            real_tensor_sfa,
                            real_tensor_sfb,
                        ),
                        (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                        (
                            tensormap_a_gmem_ptr,
                            tensormap_b_gmem_ptr,
                            tensormap_sfa_gmem_ptr,
                            tensormap_sfb_gmem_ptr,
                        ),
                        self.tma_warp_id,
                        (
                            tensormap_a_smem_ptr,
                            tensormap_b_smem_ptr,
                            tensormap_sfa_smem_ptr,
                            tensormap_sfb_smem_ptr,
                        ),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cur_k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                if is_group_changed:
                    tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_sfa_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_sfb_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc):
                    if is_leader_cta:
                        clc_pipeline.producer_acquire(clc_producer_state)
                        tile_sched.advance_to_next_work(
                            clc_pipeline.producer_get_barrier(clc_producer_state)
                        )
                        clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            # Ensure TMA warp arrives at tensormap init barrier even with no tiles,
            # otherwise the MMA warp (which arrives unconditionally) will deadlock.
            if tensormap_init_done == False:
                self.tensormap_ab_init_barrier.arrive_and_wait()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

            if cutlass.const_expr(self.use_clc):
                if is_leader_cta:
                    clc_pipeline.producer_tail(clc_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Initialize tensormaps for A, B, SFA and SFB
            #
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfa, tensormap_sfa_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfb, tensormap_sfb_smem_ptr, self.mma_warp_id
            )
            # indicate tensormap initialization has finished
            self.tensormap_ab_init_barrier.arrive_and_wait()

            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            clc_consumer_state = None
            if cutlass.const_expr(self.use_clc):
                clc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    1,
                )

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < cur_k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(cur_k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            # initialize tensorap for C
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_smem_ptr,
                self.epilog_warp_id[0],
            )
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            ### Start from here
            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            clc_consumer_state = None
            tiles_executed = cutlass.Int32(0)
            if cutlass.const_expr(self.use_clc):
                clc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    1,
                )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )
            last_group_idx = cutlass.Int32(-1)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx

                if is_group_changed:
                    # construct tensor c based on real shape, stride information
                    real_tensor_c = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,  # 2 for tensor C
                    )
                    tidx, _, _ = cute.arch.thread_idx()
                    tensormap_manager.update_tensormap(
                        ((real_tensor_c),),
                        ((tma_atom_c),),
                        ((tensormap_c_gmem_ptr),),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cur_k_tile_cnt == 0:
                        acc_vec = cute.zeros_like(acc_vec)
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_gmem_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                tiles_executed += 1

                #
                # Advance to next tile
                #
                if cutlass.const_expr(self.use_clc):
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    @cute.jit
    def make_tensor_abc_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        """Extract stride and tensor address for a given group and construct a global tensor for A, B or C.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing A, B, or C for the current group being processed, using the
        group-specific address, shape, and stride information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param strides_abc: Tensor containing strides for A, B, C for all groups. Layout: (group_count, 3, 2).
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing global memory addresses for A, B, C for all groups. Layout: (group_count, 3).
        :type tensor_address_abc: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for A, 1 for B, 2 for C.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (A, B, or C) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    @cute.jit
    def make_tensor_sfasfb_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tensor_address_sfasfb: cute.Tensor,
        tensor_index: int,
    ):
        """Extract tensor address for a given group and construct a global tensor for SFA or SFB.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing SFA or SFB for the current group being processed, using the
        group-specific address, shape information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param tensor_address_sfasfb: Tensor containing global memory addresses for SFA, SFB for all groups. Layout: (group_count, 2).
        :type tensor_address_sfasfb: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for SFA, 1 for SFB.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (SFA, SFB) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = tensor_address_sfasfb[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        c1 = cutlass.Int32(1)
        if cutlass.const_expr(tensor_index == 0):  # tensor SFA
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (m, k, c1), self.sf_vec_size
            )
            return cute.make_tensor(
                tensor_gmem_ptr,
                sfa_layout,
            )
        else:  # tensor SFB
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (n, k, c1), self.sf_vec_size
            )
            return cute.make_tensor(
                tensor_gmem_ptr,
                sfb_layout,
            )

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_tile_sched(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        use_clc: bool = False,
    ):
        """Compute tile scheduler parameters for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param use_clc: Whether to use CLC dynamic persistent tile scheduler.
        :type use_clc: bool
        """
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        if use_clc:
            return utils.ClcDynamicPersistentTileSchedulerParams(
                problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
            )
        return utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )

    @staticmethod
    def _compute_grid(
        tile_sched_params,
        max_active_clusters: int,
        use_clc: bool = False,
    ) -> tuple[int, int, int]:
        """Compute grid shape for grouped GEMM operations.

        :param tile_sched_params: Parameters for the persistent tile scheduler.
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]
        :param use_clc: Whether to use CLC dynamic persistent tile scheduler.
        :type use_clc: bool

        :return: grid: Grid shape for kernel launch.
        :rtype: tuple[int, ...]
        """
        if use_clc:
            # Standard CLC grid is (cluster_x, cluster_y, total_clusters) which
            # puts the tile count in z.  CUDA grid z is limited to 65535, so
            # move the tile count into x (limit 2^31-1):
            #   (total_clusters * cluster_x, cluster_y, 1)
            grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
                tile_sched_params
            )
            return (grid[2] * grid[0], grid[1], 1)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        cute.testing.assert_(
            grid[2] <= 65535, "Non-CLC grid z dimension exceeds CUDA limit 65535."
        )
        return grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    @cute.kernel
    def prepare_kernel(
        self,
        # Input
        input_ptr_a: cute.Pointer,
        input_ptr_b: cute.Pointer,
        input_ptr_c: cute.Pointer,
        input_ptr_sfa: cute.Pointer,
        input_ptr_sfb: cute.Pointer,
        offs: cute.Tensor,
        a_offs: cute.Tensor | None,
        b_offs: cute.Tensor | None,
        out_offset: cute.Tensor | None,
        input_problem_sizes_gmnk: tuple[int, int, int, int],
        input_strides_abc: tuple[
            tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
        ],
        dynamic_dims: tuple[int, int],
        # Output
        output_problem_sizes_mnkl: cute.Tensor,
        output_strides_abc: cute.Tensor,
        output_ptrs_abc: cute.Tensor,
        output_ptrs_sfasfb: cute.Tensor,
        output_total_num_clusters: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        if tidx == 0:
            G, M, N, K = input_problem_sizes_gmnk

            (
                (stride_am, stride_ak),
                (stride_bn, stride_bk),
                (stride_cm, stride_cn),
                (stride_bg, stride_cg),
            ) = input_strides_abc

            block_size_m = 128 * self.cluster_shape_mn[0]  # similar to estimate
            block_size_n = self.mma_tiler[1] * self.cluster_shape_mn[1]

            element_size_a = input_ptr_a.dtype.width // 8
            element_size_b = input_ptr_b.dtype.width // 8
            element_size_c = input_ptr_c.dtype.width // 8
            element_size_sf = input_ptr_sfa.dtype.width // 8

            total_num_clusters = cutlass.Int32(0)
            sf_block_size = 32

            # initializing like this is needed to be able
            # to modify variables in the control flow body
            m = cutlass.Int32(M)
            n = cutlass.Int32(N)
            k = cutlass.Int32(K)

            if K < 0:
                total_num_clusters = (
                    G
                    * ((m + block_size_m - 1) // block_size_m)
                    * ((n + block_size_n - 1) // block_size_n)
                )

            for g in range(G):
                if bidx == g:
                    off = cutlass.Int32(0)
                    end_off = offs[g]
                    if g != 0:
                        off = offs[g - 1]
                        cute.testing.assert_(
                            off % 128 == 0 and off >= 0,
                            "dynamic sizes have to be multiple of 128",
                        )
                    # Use Int64 for offset multiplications to prevent overflow
                    # when off * stride * element_size > 2^31
                    off_i64 = cutlass.Int64(off)
                    if M < 0:
                        cute.testing.assert_(
                            end_off <= dynamic_dims[0],
                            "offset is larger than A tensor size",
                        )
                        output_ptrs_abc[g, 0] = (
                            input_ptr_a.toint() + off_i64 * stride_am * element_size_a
                        )
                        output_ptrs_abc[g, 1] = (
                            input_ptr_b.toint()
                            + cutlass.Int64(g) * stride_bg * element_size_b
                        )
                        c_offset = (
                            input_ptr_c.toint() + off_i64 * stride_cm * element_size_c
                        )
                        if cutlass.const_expr(out_offset is not None):
                            c_offset += out_offset[0] * stride_cm * element_size_c
                            cute.testing.assert_(
                                end_off + out_offset[0] <= dynamic_dims[1],
                                "out_offset + offs[-1] exceeds output tensor bounds",
                            )
                        output_ptrs_abc[g, 2] = c_offset
                        output_ptrs_sfasfb[g, 0] = (
                            input_ptr_sfa.toint()
                            + off_i64 * K // sf_block_size * element_size_sf
                        )
                        output_ptrs_sfasfb[g, 1] = (
                            input_ptr_sfb.toint()
                            + cutlass.Int64(g)
                            * N
                            * K
                            // sf_block_size
                            * element_size_sf
                        )
                        m = end_off - off
                    elif K < 0:
                        cute.testing.assert_(
                            end_off <= dynamic_dims[0],
                            "offset is larger than tensor size",
                        )
                        cute.testing.assert_(
                            end_off <= dynamic_dims[1],
                            "offset is larger than tensor size",
                        )
                        a_offset = (
                            input_ptr_a.toint() + off_i64 * stride_ak * element_size_a
                        )
                        b_offset = (
                            input_ptr_b.toint() + off_i64 * stride_bk * element_size_b
                        )
                        output_ptrs_abc[g, 2] = (
                            input_ptr_c.toint()
                            + cutlass.Int64(g) * stride_cg * element_size_c
                        )
                        a_sf_offset = (
                            input_ptr_sfa.toint()
                            + off * 128 // sf_block_size * element_size_sf
                        )
                        b_sf_offset = (
                            input_ptr_sfb.toint()
                            + off * 128 // sf_block_size * element_size_sf
                        )
                        if cutlass.const_expr(a_offs is not None):
                            a_offset += a_offs[0] * stride_ak * element_size_sf
                            a_sf_offset += (
                                a_offs[0] * 128 // sf_block_size * element_size_sf
                            )
                        if cutlass.const_expr(b_offs is not None):
                            b_offset += b_offs[0] * stride_bk * element_size_sf
                            b_sf_offset += (
                                b_offs[0] * 128 // sf_block_size * element_size_sf
                            )

                        output_ptrs_abc[g, 0] = a_offset
                        output_ptrs_abc[g, 1] = b_offset
                        output_ptrs_sfasfb[g, 0] = a_sf_offset
                        output_ptrs_sfasfb[g, 1] = b_sf_offset

                        k = end_off - off
                    output_problem_sizes_mnkl[g, 0] = m
                    output_problem_sizes_mnkl[g, 1] = n
                    output_problem_sizes_mnkl[g, 2] = k
                    output_problem_sizes_mnkl[g, 3] = 1
                    output_strides_abc[g, 0, 0] = stride_am
                    output_strides_abc[g, 0, 1] = stride_ak
                    output_strides_abc[g, 1, 0] = stride_bn
                    output_strides_abc[g, 1, 1] = stride_bk
                    output_strides_abc[g, 2, 0] = stride_cm
                    output_strides_abc[g, 2, 1] = stride_cn

                if M < 0:
                    off = cutlass.Int32(0)
                    if g != 0:
                        off = offs[g - 1]
                    m = offs[g] - off

                    total_num_clusters += ((m + block_size_m - 1) // block_size_m) * (
                        (n + block_size_n - 1) // block_size_n
                    )

            if bidx == 0:
                output_total_num_clusters[0] = total_num_clusters

    # Size of smem we reserved for mbarrier, tensor memory management and tensormap update
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 5
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


_HARDWARE_INFO: utils.HardwareInfo | None = None
_SM_COUNT: int | None = None
_MAX_CLUSTERS: dict[int, int] = {}
_CACHE_KEY_TYPE = tuple[
    tuple[torch.dtype, ...],
    tuple[int, int, int],  # leading dims
    tuple[int, int],  # a and b ranks
    tuple[bool, bool, bool],  # a_offs, b_offs, out_offset are absent
    tuple[int, int, int, int],  # gmnk with -1 for dynamic dim
    tuple[int, int],  # runtime dynamic dims
    bool,  # addmm
    bool,  # use_clc
]
_COMPILED_KERNEL_CACHE: dict[
    _CACHE_KEY_TYPE,
    cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction,
] = {}


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def grouped_gemm(
    a,
    b,
    scale_a,
    scale_b,
    offs,
    *,
    addmm: bool = False,
    a_offs: torch.Tensor | None = None,
    b_offs: torch.Tensor | None = None,
    out_offset: torch.Tensor | None = None,
    num_sms: int | None = None,
    self: torch.Tensor | None = None,
    out_dtype=None,
):
    G = offs.shape[0]

    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert scale_a.dtype == scale_b.dtype == torch.float8_e8m0fnu
    for start_off in (a_offs, b_offs):
        if start_off is not None:
            assert start_off.numel() == 1
            # assert start_off.dtype == torch.int64
    if out_offset is not None:
        assert out_offset.numel() == 1
        assert b.ndim == 3, "out_offset is only supported for 2d A x 3d B (M < 0 case)"
    assert a.ndim == 2
    assert b.ndim == 2 or b.ndim == 3
    if b.ndim == 3:
        assert b.shape[0] == G

    if a.ndim == 3:
        assert a.shape[0] == G

    out_3d = a.ndim == b.ndim
    if not out_3d:
        assert a.shape[-1] == b.shape[-2]
    sf_vec_size = 32 if a.dtype == torch.float8_e4m3fn else 16
    # Note: blocked scale tensors may be over-allocated for worst-case
    # padding, so scale_a/scale_b numel can exceed a/b numel // sf_vec_size.
    # The kernel only reads the actual scale data and ignores trailing padding.

    M = a.shape[-2]
    N = b.shape[-1]
    K = a.shape[-1]
    if out_3d:
        assert a.shape[-2] % 128 == 0, (
            "for 2d-2d inputs, a non-contraction dimension has to be multiple of 128"
        )
        assert b.shape[-1] % 128 == 0, (
            "for 2d-2d inputs, b non-contraction dimension has to be multiple of 128"
        )
    else:
        assert a.shape[-1] % 128 == 0, (
            "for 3d-2d inputs, a contraction dimension has to be multiple of 128"
        )
        assert b.shape[-1] % 128 == 0, (
            "for 3d-2d inputs, b non-contraction dimension has to be multiple of 128"
        )

    problem_sizes_mnk = (M, N, -1) if out_3d else (-1, N, K)

    # the kernel expects b to have contraction dimension as the last,
    # do that to minimize kernel changes
    b = b.transpose(-2, -1)

    device = a.device
    out_shape = (G, M, N) if out_3d else (M, N)
    if self is not None:
        if out_offset is not None:
            assert not out_3d
            assert self.shape[0] >= M
            assert self.shape[1] == N
        else:
            assert self.shape == torch.Size(out_shape), (
                f"Provided output shape {self.shape} does not match expected {out_shape}"
            )
        assert self.dtype in (torch.bfloat16, torch.float32)
        assert out_dtype is None or self.dtype == out_dtype
        out = self
    else:
        out = torch.empty(
            out_shape,
            dtype=out_dtype if out_dtype is not None else torch.bfloat16,
            device=device,
        )

    # if not set we will use by default if num_sms is not set, use env-var to force
    clc_env = os.environ.get("GROUPED_GEMM_CLC")
    if clc_env is None:
        use_clc = num_sms is None
    else:
        assert clc_env in ("0", "1"), "GROUPED_GEMM_CLC must be '0' or '1'"
        use_clc = clc_env == "1"

    # configuration (might need to tune depending on statically known args)
    mma_tiler_mn: tuple[int, int] = (256, 256)
    cluster_shape_mn: tuple[int, int] = (2, 1)

    grouped_blockscaled_gemm = Sm100GroupedBlockScaledGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        use_clc=use_clc,
    )

    def ld(x: torch.Tensor):
        strides = x.stride()
        ndim = len(strides)
        for i in range(ndim - 1, max(ndim - 3, -1), -1):
            if strides[i] == 1:
                return i
        raise ValueError(
            f"Neither stride[-1]={strides[-1]} nor stride[-2]={strides[-2]} is 1"
        )

    b_arg = b if b.ndim == 2 else b[0]
    out_arg = out if out.ndim == 2 else out[0]
    leading_dims: tuple[int, ...] = tuple([ld(x) for x in (a, b_arg, out_arg)])
    cute_tensors_abc: list[cute.Tensor] = [
        from_dlpack(x.unsqueeze(-1).detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=ld
        )
        for x, ld in zip((a, b_arg, out_arg), leading_dims)
    ]

    strides_abc = [(x.stride(-2), x.stride(-1)) for x in (a, b, out)]
    stride_bg = b.stride(0) if b.ndim == 3 else -1
    stride_cg = out.stride(0) if out.ndim == 3 else -1
    strides_abc.append((stride_bg, stride_cg))
    strides_abc = tuple(strides_abc)

    cute_tensors_sfasfb: list[cute.Tensor] = [
        from_dlpack(x.detach(), assumed_align=16) for x in (scale_a, scale_b)
    ]
    cute_offs: cute.Tensor = from_dlpack(offs)
    # Workaround CuteDSL bug that could not infer dtype of scalar tensors by turning it into 1D tensor
    cute_start_offs: list[cute.Tensor | None] = [
        x if x is None else from_dlpack(x.view(-1).detach()) for x in (a_offs, b_offs)
    ]
    cute_out_offset: cute.Tensor | None = (
        None if out_offset is None else from_dlpack(out_offset.view(-1).detach())
    )

    torch_num_total_clusters: torch.Tensor = torch.empty(
        (1,), dtype=torch.int32, device=device
    )
    cute_num_total_clusters: cute.Tensor = from_dlpack(
        torch_num_total_clusters, assumed_align=16
    )

    global _HARDWARE_INFO
    global _SM_COUNT
    hardware_info: utils.HardwareInfo
    if _HARDWARE_INFO is None:
        hardware_info = utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        _HARDWARE_INFO = hardware_info
        _SM_COUNT = sm_count
    else:
        hardware_info = _HARDWARE_INFO
        sm_count = _SM_COUNT

    WARP_SIZE = 32
    G_padded = max(G, WARP_SIZE)

    torch_problem_sizes_mnkl: torch.Tensor = torch.zeros(
        (G_padded, 4), dtype=torch.int32, device=device
    )
    cute_problem_sizes_mnkl: cute.Tensor = from_dlpack(
        torch_problem_sizes_mnkl, assumed_align=16
    )

    torch_ptrs_abc: torch.Tensor = torch.zeros(
        (G_padded, 3), dtype=torch.int64, device=device
    )
    cute_ptrs_abc: cute.Tensor = from_dlpack(torch_ptrs_abc, assumed_align=16)

    torch_ptrs_sfasfb: torch.Tensor = torch.zeros(
        (G_padded, 2), dtype=torch.int64, device=device
    )
    cute_ptrs_sfasfb: cute.Tensor = from_dlpack(torch_ptrs_sfasfb, assumed_align=16)

    torch_strides_abc: torch.Tensor = torch.zeros(
        (G_padded, 3, 2), dtype=torch.int32, device=device
    )
    cute_strides_abc: cute.Tensor = from_dlpack(torch_strides_abc, assumed_align=16)

    current_stream: cuda.CUstream = cutlass_torch.current_stream()

    def estimate_total_num_clusters(
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        out_shape: tuple[int, int, int] | tuple[int, int],
        offs: torch.Tensor,
    ) -> int:
        block_size_m = (
            128 * cluster_shape_mn[0]
        )  # either mma_tiler is 128, or we are using 2cta instrs
        block_size_n = mma_tiler_mn[1] * cluster_shape_mn[1]
        if len(out_shape) == 2:
            num_groups = offs.numel()
            # equivalent to adding overestimating each group by 1
            tiles_m_upper_bound = cdiv(out_shape[-2], block_size_m) + num_groups - 1
            return tiles_m_upper_bound * cdiv(out_shape[-1], block_size_n)

        return (
            cdiv(out_shape[-2], block_size_m)
            * cdiv(out_shape[-1], block_size_n)
            * out_shape[0]
        )

    est_total_num_clusters = estimate_total_num_clusters(
        mma_tiler_mn, cluster_shape_mn, out_shape, offs
    )

    if use_clc:
        # :think: We might be able to use smid to index into the num_sm_count but feels sketch
        num_tensormap_buffers = (
            est_total_num_clusters * cluster_shape_mn[0] * cluster_shape_mn[1]
        )
    else:
        num_tensormap_buffers = sm_count

    tensormap_shape = (
        num_tensormap_buffers,
        Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
        Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
    )
    torch_tensor_of_tensormap: torch.Tensor = torch.empty(
        tensormap_shape, dtype=torch.int64, device=device
    )
    cute_tensor_of_tensormap: cute.Tensor = from_dlpack(
        torch_tensor_of_tensormap, assumed_align=16
    )

    dynamic_dims = (a.shape[-1], b.shape[-1]) if out_3d else (a.shape[0], out.shape[-2])
    cache_key = (
        (a.dtype, b.dtype, out.dtype),
        leading_dims,
        (a.ndim, b.ndim),
        (a_offs is None, b_offs is None, out_offset is None),
        (G, *problem_sizes_mnk),
        dynamic_dims,
        addmm,
        use_clc,
    )
    cluster_shape = cluster_shape_mn[0] * cluster_shape_mn[1]
    tile_sf_to_dynamic_dim = out_3d

    if cache_key not in _COMPILED_KERNEL_CACHE:
        if cluster_shape not in _MAX_CLUSTERS:
            _MAX_CLUSTERS[cluster_shape] = hardware_info.get_max_active_clusters(
                cluster_shape
            )
        max_active_clusters = _MAX_CLUSTERS[cluster_shape]

        compiled_grouped_gemm = cute.compile(
            grouped_blockscaled_gemm,
            cute_tensors_abc[0],
            cute_tensors_abc[1],
            cute_tensors_abc[2],
            cute_tensors_sfasfb[0],
            cute_tensors_sfasfb[1],
            cute_offs,
            cute_start_offs[0],
            cute_start_offs[1],
            cute_out_offset,
            cute_num_total_clusters,
            cute_problem_sizes_mnkl,
            cute_strides_abc,
            cute_ptrs_abc,
            cute_ptrs_sfasfb,
            est_total_num_clusters,
            cute_tensor_of_tensormap,
            max_active_clusters,
            strides_abc,
            tile_sf_to_dynamic_dim,
            dynamic_dims,
            G,
            problem_sizes_mnk[0],
            problem_sizes_mnk[1],
            problem_sizes_mnk[2],
            addmm,
            current_stream,
            options="--opt-level 2 --enable-assertions",
        )
        _COMPILED_KERNEL_CACHE[cache_key] = compiled_grouped_gemm

    fn = _COMPILED_KERNEL_CACHE[cache_key]

    if num_sms is not None:
        max_active_clusters = num_sms // cluster_shape
        assert max_active_clusters <= _MAX_CLUSTERS[cluster_shape]
    else:
        max_active_clusters = _MAX_CLUSTERS[cluster_shape]

    fn(
        cute_tensors_abc[0],
        cute_tensors_abc[1],
        cute_tensors_abc[2],
        cute_tensors_sfasfb[0],
        cute_tensors_sfasfb[1],
        cute_offs,
        cute_start_offs[0],
        cute_start_offs[1],
        cute_out_offset,
        cute_num_total_clusters,
        cute_problem_sizes_mnkl,
        cute_strides_abc,
        cute_ptrs_abc,
        cute_ptrs_sfasfb,
        cute_tensor_of_tensormap,
        max_active_clusters,
        strides_abc,
        dynamic_dims,
        current_stream,
    )

    return out


def grouped_addmm_(
    out,
    a,
    b,
    scale_a,
    scale_b,
    offs,
    *,
    a_offs=None,
    b_offs=None,
    out_offset=None,
    num_sms=None,
    out_dtype=None,
):
    return grouped_gemm(
        a,
        b,
        scale_a,
        scale_b,
        offs,
        a_offs=a_offs,
        b_offs=b_offs,
        out_offset=out_offset,
        num_sms=num_sms,
        self=out,
        out_dtype=out_dtype,
        addmm=True,
    )


if __name__ == "__main__":

    def prep_inputs(M_pad, E, K, N, out_3d=False, return_out=False):
        assert K % 128 == 0
        assert N % 128 == 0
        per_expert = M_pad // E
        offs = torch.arange(
            per_expert, M_pad + 1, per_expert, device="cuda", dtype=torch.int32
        )
        sf_vec_size = 32
        out = None
        if not out_3d:
            sf_k = K // sf_vec_size
            A = torch.randn(M_pad, K, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            B = (
                torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16)
                .to(torch.float8_e4m3fn)
                .transpose(-2, -1)
            )
            A_scales = torch.ones(M_pad, sf_k, device="cuda", dtype=torch.uint8).to(
                torch.float8_e8m0fnu
            )
            B_scales = torch.ones(E, N * sf_k, device="cuda", dtype=torch.uint8).to(
                torch.float8_e8m0fnu
            )
            if return_out:
                out = torch.randn((M_pad, N), device="cuda", dtype=torch.float32)
        else:
            sf_k = M_pad // sf_vec_size
            A = (
                torch.randn(M_pad, N, device="cuda", dtype=torch.bfloat16)
                .to(torch.float8_e4m3fn)
                .t()
            )
            B = torch.randn(M_pad, K, device="cuda", dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            )
            A_scales = torch.full((N, sf_k), float("nan"), device="cuda").to(
                torch.float8_e8m0fnu
            )
            B_scales = torch.full((K, sf_k), 1.0, device="cuda").to(
                torch.float8_e8m0fnu
            )
            offs[-1] -= 128
            print(offs)
            offlist = offs.tolist()
            N_tiles = N // 128
            K_tiles = K // 128
            M_tiles = sf_k // (128 // sf_vec_size)
            A_sf_view = A_scales.view(N_tiles, M_tiles, -1)
            B_sf_view = B_scales.view(K_tiles, M_tiles, -1)
            start_tile = 0
            for i in range(len(offlist)):
                val = torch.tensor([2**i], device="cuda", dtype=torch.float).to(
                    torch.float8_e4m3fn
                )
                end_tile = offlist[i] // 128
                A_sf_view[:, start_tile:end_tile, :].copy_(val)
                start_tile = end_tile
            if return_out:
                out = torch.full((E, N, K), 20.0, device="cuda", dtype=torch.float32)

        return A, A_scales, B, B_scales, offs, out

    K = 7168
    N = 4096
    per_expert = 4096
    E = 4
    M_pad = per_expert * E
    out_3d = False
    A_q, A_scales, W, W_scales, offs, out = prep_inputs(
        M_pad, E, K, N, out_3d=out_3d, return_out=True
    )
    offs1 = offs.clone()
    offs1[0] += 128
    # offs1[3] -= 128
    # out_t = torch._scaled_grouped_mm(A_q, W, A_scales, W_scales, offs=offs)

    def verify(x, w, offs, out, include_sf=False):
        offs = offs.tolist()
        start_idx = 0
        x = x.to(torch.bfloat16)
        w = w.to(torch.bfloat16)
        for i in range(len(offs)):
            sf = 2**i if include_sf else 1
            end_idx = offs[i]
            if out.ndim == 2:
                x_i = x[start_idx:end_idx]
                w_i = w[i]
                out_slice = out[start_idx:end_idx]
            else:
                x_i = x[:, start_idx:end_idx]
                w_i = w[start_idx:end_idx, :]
                out_slice = out[i]
            out_i = (x_i @ w_i) * sf
            print(i, (out_i - out_slice).abs().max())
            # print(out_slice[:256, 0])
            start_idx = end_idx

    init = out.clone()
    out = grouped_gemm(A_q, W, A_scales, W_scales, offs, self=out) - init

    # out_t = torch._scaled_grouped_mm(A_q, W, A_scales, W_scales, offs=offs)
    verify(A_q, W, offs, out, include_sf=out_3d)
    with torch.profiler.profile(with_stack=True, record_shapes=True) as prof:
        for _ in range(4):
            out = grouped_gemm(A_q, W, A_scales, W_scales, offs1, self=out)
    for e in prof.events():
        if "kernel_cutlass" in e.name:
            print(e.name, e.device_time)
    # prof.export_chrome_trace("/tmp/grouped_gemm_trace.json")
    # out_t = torch._scaled_grouped_mm(A_q, W, A_scales, W_scales, offs=offs1)
    # verify(A_q, W, offs1, out, True)

    print("PASS")
