# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
import importlib.util
from typing import Tuple

import torch

from torchao.utils import ceil_div

_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list[str]:
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        if (
            importlib.util.find_spec(module_name) is None
            and package_name not in missing
        ):
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    return len(_missing_cutedsl_runtime_packages()) == 0


def _make_tile_smem_layouts(cute, tile_m: int, tile_k: int):
    smem_layout_in = cute.make_layout(
        (tile_m, tile_k),
        stride=(tile_k, 1),
    )
    smem_layout_out = cute.make_layout(
        (tile_m, tile_k),
        stride=(tile_k, 1),
    )
    return smem_layout_in, smem_layout_out


# Config format:
# (compute_warps, tile_m, tile_k, k_tiles_per_cta)
_CUTEDSL_CONFIGS = {
    "bf16_default": (4, 128, 32, 4),
    "fallback": (6, 128, 32, 2),
}


def _select_cutedsl_config(
    input_dtype_name: str,
    scaling_mode: str,
    K: int,
) -> Tuple[str, Tuple[int, int, int, int]]:
    del K

    if input_dtype_name == "torch.bfloat16":
        config_name = "bf16_default"
    else:
        config_name = "fallback"
    return config_name, _CUTEDSL_CONFIGS[config_name]


@functools.cache
def _compile_mxfp8_quantize_2d_cutedsl(
    input_dtype_name: str,
    scaling_mode: str,
    compute_warps: int,
    tile_m: int,
    tile_k: int,
    requested_stage_count: int,
    k_tiles_per_cta: int,
    is_full_k_tiles: bool,
    is_blackwell: bool,
    blocked_scale_output: bool,
    use_group_alignment: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass._mlir.dialects import llvm
    from cutlass.base_dsl._mlir_helpers import arith as _dsl_arith
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.cutlass_dsl import T, dsl_user_op

    # PTX lowering note:
    # - RCEIL uses inline PTX on Blackwell-family targets because
    #   CuTeDSL does not currently lower this conversion to
    #   `cvt.rp.satfinite.ue8m0x2.f32` on its own.
    # - FLOOR still uses a different lowered sequence than C++
    #   helper routines.

    if input_dtype_name == "torch.float32":
        INPUT_CUTLASS_DTYPE = cutlass.Float32
    elif input_dtype_name == "torch.bfloat16":
        INPUT_CUTLASS_DTYPE = cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported input dtype for CuTeDSL quantize_2d: {input_dtype_name}"
        )

    # Warp-specialized TMA kernel:
    # - warp 0: producer (issues TMA G2S and S2G)
    # - warps [1..compute_warps]: consumers (quantize)
    # Note: we intentionally keep store on warp 0 (no dedicated store
    # warp).  A split load-warp/store-warp design was tested and
    # mostly regressed throughput, so this layout is the tuned
    # default.
    COMPUTE_WARPS = compute_warps
    TILE_M = tile_m
    TILE_K = tile_k
    K_TILES_PER_CTA = k_tiles_per_cta
    IS_FULL_K_TILES_VALUE = is_full_k_tiles
    IS_BLACKWELL_VALUE = is_blackwell
    BLOCKED_SCALE_OUTPUT_VALUE = blocked_scale_output
    USE_GROUP_ALIGNMENT_VALUE = use_group_alignment

    THREADS_PER_BLOCK = (1 + COMPUTE_WARPS) * 32
    assert COMPUTE_WARPS >= 1
    assert TILE_M > 0 and TILE_K > 0
    assert TILE_K % 32 == 0

    SCALE_DIM_K_VALUE = 32
    K_BLOCKS_PER_TILE = TILE_K // SCALE_DIM_K_VALUE
    assert K_BLOCKS_PER_TILE > 0
    assert requested_stage_count >= 1
    # B200 sweeps on our representative 3D shapes showed no benefit
    # beyond 2 stages. We keep stage setup generic so future tuning can
    # revisit this, but the current tuned contract is 1 or 2 stages.
    assert requested_stage_count <= 2
    assert K_TILES_PER_CTA >= 1
    STAGE_COUNT_VALUE = min(requested_stage_count, K_TILES_PER_CTA)

    input_elem_bytes = 4 if input_dtype_name == "torch.float32" else 2
    TILE_COPY_BYTES = TILE_M * TILE_K * input_elem_bytes
    M_THREADS = COMPUTE_WARPS * 32
    M_ITERS_PER_LANE = ceil_div(TILE_M, M_THREADS)

    F8_MAX = cutlass.Float32(448.0)
    INV_F8_MAX = cutlass.Float32(1.0 / 448.0)

    @dsl_user_op
    def _cvt_rp_satfinite_ue8m0x2_f32(
        a: cutlass.Float32,
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Uint16:
        return cutlass.Uint16(
            llvm.inline_asm(
                T.i16(),
                [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
                "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
                "=h,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.struct
    class SharedStorage:
        tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, STAGE_COUNT_VALUE]
        in_smem: cute.struct.Align[
            cute.struct.MemRange[
                INPUT_CUTLASS_DTYPE, STAGE_COUNT_VALUE * TILE_M * TILE_K
            ],
            128,
        ]
        out_smem: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float8E4M3FN, STAGE_COUNT_VALUE * TILE_M * TILE_K
            ],
            128,
        ]

    class Mxfp8Quantize2dKernel:
        @cute.jit
        def _compute_padded_m_index(
            self,
            m_input: cutlass.Int32,
            group_end_offsets: cute.Tensor,
            group_alignment_size: cutlass.Int32,
            num_groups: cutlass.Int32,
        ):
            """
            Map input M index to padded output M index.

            For each group, we add padding to align to group_alignment_size.
            This function computes the cumulative padding offset for all previous groups.

            Returns: (padded_m_index, is_valid)
                - padded_m_index: output M index in the padded tensor
                - is_valid: True if this is a real token (not padding)
            """
            group_start = cutlass.Int32(0)
            cumulative_padding = cutlass.Int32(0)
            padded_m = cutlass.Int32(0)
            is_valid = False

            for g in range(num_groups):
                group_end = cutlass.Int32(group_end_offsets[g])
                group_size = group_end - group_start

                # Check if m_input is in this group
                in_this_group = m_input < group_end
                if in_this_group and not is_valid:
                    # This input M belongs to this group
                    padded_m = m_input + cumulative_padding
                    is_valid = True

                # Calculate padding for this group (always compute to maintain loop structure)
                aligned_size = ((group_size + group_alignment_size - 1) // group_alignment_size) * group_alignment_size
                padding_for_group = aligned_size - group_size
                cumulative_padding += padding_for_group
                group_start = group_end

            return padded_m, is_valid

        @cute.jit
        def _load_vals_block_full(
            self,
            sIN_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
        ):
            vals_block = cute.make_rmem_tensor((SCALE_DIM_K_VALUE,), cutlass.Float32)
            for i in range(SCALE_DIM_K_VALUE):
                vals_block[i] = cutlass.Float32(sIN_tile[m_rel, k_base + i])
            return vals_block

        @cute.jit
        def _load_vals_block_tail(
            self,
            sIN_tile: cute.Tensor,
            k0: cutlass.Int64,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
            K: cutlass.Int64,
        ):
            vals_block = cute.make_rmem_tensor((SCALE_DIM_K_VALUE,), cutlass.Float32)
            for i in range(SCALE_DIM_K_VALUE):
                k = k0 + k_base + i
                if k < K:
                    vals_block[i] = cutlass.Float32(sIN_tile[m_rel, k_base + i])
                else:
                    vals_block[i] = cutlass.Float32(0.0)
            return vals_block

        @cute.jit
        def _compute_amax(self, vals_block: cute.Tensor):
            vals_vec = vals_block.load()
            abs_vec = cute.where(vals_vec < 0, -vals_vec, vals_vec)
            return cutlass.Float32(
                abs_vec.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
            )

        @cute.jit
        def _compute_scale_rceil(
            self,
            amax: cutlass.Float32,
        ):
            descale = amax * INV_F8_MAX
            if cutlass.const_expr(IS_BLACKWELL_VALUE):
                scale_biased = cutlass.Int32(_cvt_rp_satfinite_ue8m0x2_f32(descale))
                inv_scale = cutlass.Float32(1.0)
                if scale_biased == 0xFF:
                    inv_scale = cutlass.Float32(0.0)
                elif scale_biased == 0:
                    inv_scale = cute.exp2(cutlass.Float32(126.0))
                else:
                    inv_scale = cute.exp2(cutlass.Float32(127 - scale_biased))
                return scale_biased, inv_scale

            bits = _dsl_arith.bitcast(descale.ir_value(), _dsl_arith.T.i32())
            exponent = (bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
            mantissa = bits & cutlass.Int32(0x7FFFFF)
            if exponent == 0xFF:
                if mantissa != 0:
                    scale_biased = cutlass.Int32(0xFF)
                else:
                    scale_biased = cutlass.Int32(0xFE)
            else:
                if mantissa > 0:
                    if exponent != 0xFE:
                        if exponent == 0:
                            if mantissa > 0x400000:
                                exponent += 1
                        else:
                            exponent += 1
                scale_biased = exponent

            inv_scale = cutlass.Float32(1.0)
            if scale_biased == 0xFF:
                inv_scale = cutlass.Float32(0.0)
            elif scale_biased == 0:
                inv_scale = cutlass.Float32(1.0)
            else:
                inv_scale = cute.exp2(cutlass.Float32(127 - scale_biased))
            return scale_biased, inv_scale

        @cute.jit
        def _compute_scale_floor(
            self,
            amax: cutlass.Float32,
        ):
            bits = _dsl_arith.bitcast(amax.ir_value(), _dsl_arith.T.i32())
            exp_i = ((bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)) - cutlass.Int32(
                127
            )
            scale_exp_unbiased = exp_i - cutlass.Int32(8)
            if scale_exp_unbiased < -127:
                scale_exp_unbiased = cutlass.Int32(-127)
            if scale_exp_unbiased > 128:
                scale_exp_unbiased = cutlass.Int32(128)
            inv_scale = cute.exp2(cutlass.Float32(-scale_exp_unbiased))
            scale_biased = scale_exp_unbiased + 127
            return scale_biased, inv_scale

        @cute.jit
        def _compute_scale_from_amax(
            self,
            amax: cutlass.Float32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            scale_biased = cutlass.Int32(0)
            inv_scale = cutlass.Float32(1.0)
            if amax > 0:
                if cutlass.const_expr(USE_RCEIL):
                    scale_biased, inv_scale = self._compute_scale_rceil(amax)
                else:
                    scale_biased, inv_scale = self._compute_scale_floor(amax)
            return scale_biased, inv_scale

        @cute.jit
        def _store_scales_vectorized(
            self,
            scales_tensor: cute.Tensor,
            m: cutlass.Int64,
            k_block_base: cutlass.Int64,
            scale_buffer: cute.Tensor,
            num_scales: cutlass.Int32,
            BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
        ):
            """Store scales using vectorized 4-byte writes when num_scales=4."""
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
                # Blocked layout with 4 contiguous scales - write as uint32
                if num_scales == 4:
                    # Pack 4 uint8 scales into uint32 and write
                    scales_tensor_u32 = cute.recast_tensor(
                        scales_tensor, cutlass.Uint32
                    )
                    scale_buffer_u32 = cute.recast_tensor(scale_buffer, cutlass.Uint32)
                    scales_tensor_u32[m, k_block_base // cutlass.Int64(4)] = (
                        scale_buffer_u32[0]
                    )
                else:
                    # Fallback for non-4 cases (e.g., tail tiles)
                    for i in range(num_scales):
                        k_block = k_block_base + i
                        scales_tensor[m, k_block] = scale_buffer[i]
            else:
                # Row-major layout - scalar stores
                for i in range(num_scales):
                    k_block = k_block_base + i
                    scales_tensor[m, k_block] = scale_buffer[i]

        @cute.jit
        def _store_q_fp8_chunk(
            self,
            q_fp8_vals4: cute.Tensor,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            sout_base: cutlass.Int32,
        ):
            """Store 4 FP8 values to shared memory"""
            sOUT_tile_u32 = cute.recast_tensor(sOUT_tile, cutlass.Uint32)
            q_fp8_vals4_u32 = cute.recast_tensor(q_fp8_vals4, cutlass.Uint32)
            sOUT_tile_u32[m_rel, sout_base // cutlass.Int32(4)] = q_fp8_vals4_u32[0]

        @cute.jit
        def _load_vals_chunk_full(
            self,
            vals_block: cute.Tensor,
            local_base: cutlass.Int32,
        ):
            chunk_vec = 4
            vals_chunk = cute.make_rmem_tensor((chunk_vec,), cutlass.Float32)
            for j in range(chunk_vec):
                vals_chunk[j] = vals_block[local_base + j]
            return vals_chunk

        @cute.jit
        def _load_vals_chunk_tail(
            self,
            vals_block: cute.Tensor,
            k0: cutlass.Int64,
            sout_base: cutlass.Int32,
            local_base: cutlass.Int32,
            K: cutlass.Int64,
        ):
            chunk_vec = 4
            vals_chunk = cute.make_rmem_tensor((chunk_vec,), cutlass.Float32)
            for j in range(chunk_vec):
                k = k0 + sout_base + j
                if k < K:
                    vals_chunk[j] = vals_block[local_base + j]
                else:
                    vals_chunk[j] = cutlass.Float32(0.0)
            return vals_chunk

        @cute.jit
        def _quantize_store_chunk(
            self,
            vals_chunk: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            sout_base: cutlass.Int32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            q_vals4_vec = vals_chunk.load() * inv_scale
            if not cutlass.const_expr(USE_RCEIL):
                q_vals4_vec = cute.where(q_vals4_vec > F8_MAX, F8_MAX, q_vals4_vec)
                q_vals4_vec = cute.where(q_vals4_vec < -F8_MAX, -F8_MAX, q_vals4_vec)
            q_fp8_vec4 = q_vals4_vec.to(cutlass.Float8E4M3FN)
            q_fp8_vals4 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            q_fp8_vals4.store(q_fp8_vec4)
            self._store_q_fp8_chunk(q_fp8_vals4, sOUT_tile, m_rel, sout_base)

        @cute.jit
        def _quantize_store_full(
            self,
            vals_block: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            chunk_vec = 4
            num_chunks = SCALE_DIM_K_VALUE // chunk_vec
            for c in range(num_chunks):
                local_base = c * chunk_vec
                sout_base = k_base + local_base
                vals_chunk = self._load_vals_chunk_full(vals_block, local_base)
                self._quantize_store_chunk(
                    vals_chunk, inv_scale, sOUT_tile, m_rel, sout_base, USE_RCEIL
                )

        @cute.jit
        def _quantize_store_tail(
            self,
            vals_block: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            k0: cutlass.Int64,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
            K: cutlass.Int64,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            chunk_vec = 4
            num_chunks = SCALE_DIM_K_VALUE // chunk_vec
            for c in range(num_chunks):
                local_base = c * chunk_vec
                sout_base = k_base + local_base
                vals_chunk = self._load_vals_chunk_tail(
                    vals_block, k0, sout_base, local_base, K
                )
                self._quantize_store_chunk(
                    vals_chunk, inv_scale, sOUT_tile, m_rel, sout_base, USE_RCEIL
                )

        @cute.jit
        def _issue_tma_load(
            self,
            tma_atom_in: cute.CopyAtom,
            gIN_tile: cute.Tensor,
            sIN_tile: cute.Tensor,
            tma_mbar_ptr: cutlass.Int64,
            warp_idx: cutlass.Int32,
        ):
            if warp_idx == 0:
                cta_layout = cute.make_layout((1,))
                sIN_for_tma_partition = cute.group_modes(sIN_tile, 0, 1)
                gIN_for_tma_partition = cute.group_modes(gIN_tile, 0, 1)
                tINs, tINg = cpasync.tma_partition(
                    tma_atom_in,
                    0,
                    cta_layout,
                    sIN_for_tma_partition,
                    gIN_for_tma_partition,
                )
                tINg_stage0 = tINg[(None, 0)]
                tINs_stage0 = tINs[(None, 0)]
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        tma_mbar_ptr, TILE_COPY_BYTES
                    )
                cute.copy(
                    tma_atom_in,
                    tINg_stage0,
                    tINs_stage0,
                    tma_bar_ptr=tma_mbar_ptr,
                )

        @cute.jit
        def _issue_tma_store(
            self,
            tma_atom_out: cute.CopyAtom,
            gOUT_tile: cute.Tensor,
            sOUT_tile: cute.Tensor,
            warp_idx: cutlass.Int32,
        ):
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            cute.arch.sync_threads()
            if warp_idx == 0:
                cta_layout = cute.make_layout((1,))
                sOUT_for_tma_partition = cute.group_modes(sOUT_tile, 0, 1)
                gOUT_for_tma_partition = cute.group_modes(gOUT_tile, 0, 1)
                tOUTs, tOUTg = cpasync.tma_partition(
                    tma_atom_out,
                    0,
                    cta_layout,
                    sOUT_for_tma_partition,
                    gOUT_for_tma_partition,
                )
                tOUTs_stage0 = tOUTs[(None, 0)]
                tOUTg_stage0 = tOUTg[(None, 0)]
                cute.copy(
                    tma_atom_out,
                    tOUTs_stage0,
                    tOUTg_stage0,
                )

        @cute.jit
        def _store_output_with_padding(
            self,
            sOUT_tile: cute.Tensor,
            out_mk: cute.Tensor,
            m0_input: cutlass.Int64,
            k0: cutlass.Int64,
            group_end_offsets: cute.Tensor,
            group_alignment_size: cutlass.Int32,
            num_groups: cutlass.Int32,
            warp_idx: cutlass.Int32,
        ):
            """Store quantized output with group alignment padding."""
            cute.arch.sync_threads()
            # All threads participate in copying data from smem to gmem
            tidx, _, _ = cute.arch.thread_idx()
            num_threads_tuple = cute.arch.block_dim()
            num_threads = num_threads_tuple[0]

            tile_size = cutlass.Int32(TILE_M * TILE_K)
            for idx in range(tile_size):
                thread_owns = (idx % num_threads) == tidx
                if thread_owns:
                    m_rel = cutlass.Int32(idx // TILE_K)
                    k_rel = cutlass.Int32(idx % TILE_K)
                    m_input = m0_input + cutlass.Int64(m_rel)
                    k = k0 + cutlass.Int64(k_rel)

                    # Compute padded output index
                    m_output_i32, is_valid = self._compute_padded_m_index(
                        cutlass.Int32(m_input), group_end_offsets, group_alignment_size, num_groups
                    )
                    m_output = cutlass.Int64(m_output_i32)

                    if is_valid:
                        out_mk[m_output, k] = sOUT_tile[m_rel, k_rel]

        @cute.kernel
        def kernel(
            self,
            inp_mk: cute.Tensor,
            tma_atom_in: cute.CopyAtom,
            tma_tensor_in: cute.Tensor,
            out_mk: cute.Tensor,
            tma_atom_out: cute.CopyAtom,
            tma_tensor_out: cute.Tensor,
            scales_colwise_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            blocked_scale_layout: cute.Layout,
            group_end_offsets: cute.Tensor,
            group_alignment_size: cutlass.Int32,
            num_groups: cutlass.Int32,
            SCALE_DIM_K: cutlass.Constexpr[int],
            USE_RCEIL: cutlass.Constexpr[bool],
            IS_FULL_K_TILES: cutlass.Constexpr[bool],
            STAGE_COUNT: cutlass.Constexpr[int],
            IS_BLACKWELL: cutlass.Constexpr[bool],
            USE_GROUP_ALIGNMENT: cutlass.Constexpr[bool],
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
            bidx, bidy, _ = cute.arch.block_idx()

            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            # The tuned contract keeps STAGE_COUNT <= 2.
            tma_mbar_ptr0 = storage.tma_mbar_ptr.data_ptr()
            tma_mbar_ptr1 = tma_mbar_ptr0
            if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                tma_mbar_ptr1 = tma_mbar_ptr0 + 1

            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(
                cute, TILE_M, TILE_K
            )
            staged_layout_in = cute.make_layout(
                (STAGE_COUNT_VALUE, TILE_M, TILE_K),
                stride=(TILE_M * TILE_K, TILE_K, 1),
            )
            staged_layout_out = cute.make_layout(
                (STAGE_COUNT_VALUE, TILE_M, TILE_K),
                stride=(TILE_M * TILE_K, TILE_K, 1),
            )
            sIN_staged = storage.in_smem.get_tensor(staged_layout_in)
            sOUT_staged = storage.out_smem.get_tensor(staged_layout_out)
            stage_elems = TILE_M * TILE_K
            sIN_tile0 = cute.make_tensor(
                sIN_staged.iterator + 0 * stage_elems, smem_layout_in
            )
            sOUT_tile0 = cute.make_tensor(
                sOUT_staged.iterator + 0 * stage_elems, smem_layout_out
            )
            sIN_tile1 = sIN_tile0
            sOUT_tile1 = sOUT_tile0
            if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                sIN_tile1 = cute.make_tensor(
                    sIN_staged.iterator + 1 * stage_elems, smem_layout_in
                )
                sOUT_tile1 = cute.make_tensor(
                    sOUT_staged.iterator + 1 * stage_elems, smem_layout_out
                )

            if tidx == 0:
                cpasync.prefetch_descriptor(tma_atom_in)
                cpasync.prefetch_descriptor(tma_atom_out)
                cute.arch.mbarrier_init(tma_mbar_ptr0, 1)
                if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                    cute.arch.mbarrier_init(tma_mbar_ptr1, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            k_tile_group_idx = cutlass.Int64(bidx)
            m_tile = cutlass.Int64(bidy)
            m0 = m_tile * TILE_M
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                scales_tensor = cute.make_tensor(
                    scales_colwise_u8.iterator,
                    blocked_scale_layout,
                )
            else:
                scales_tensor = scales_colwise_u8
            for tile_step in cutlass.range_constexpr(K_TILES_PER_CTA):
                k_tile_eff = k_tile_group_idx * K_TILES_PER_CTA + tile_step
                k0 = k_tile_eff * TILE_K

                stage_idx = tile_step % STAGE_COUNT

                sIN_tile = sIN_tile0
                sOUT_tile = sOUT_tile0
                tma_mbar_ptr = tma_mbar_ptr0
                if cutlass.const_expr(STAGE_COUNT > 1):
                    tma_mbar_ptr = tma_mbar_ptr0 + stage_idx
                if cutlass.const_expr(STAGE_COUNT > 1):
                    if stage_idx == 1:
                        sIN_tile = sIN_tile1
                        sOUT_tile = sOUT_tile1

                tma_phase = (tile_step // STAGE_COUNT) % 2

                if cutlass.const_expr(
                    tile_step == 0 or not (STAGE_COUNT > 1 and K_TILES_PER_CTA > 1)
                ):
                    gIN_tile = cute.local_tile(
                        tma_tensor_in, (TILE_M, TILE_K), (m_tile, k_tile_eff)
                    )
                    self._issue_tma_load(
                        tma_atom_in,
                        gIN_tile,
                        sIN_tile,
                        tma_mbar_ptr,
                        warp_idx,
                    )

                if cutlass.const_expr(STAGE_COUNT > 1 and K_TILES_PER_CTA > 1):
                    if cutlass.const_expr(tile_step + 1 < K_TILES_PER_CTA):
                        k_tile_next = k_tile_group_idx * K_TILES_PER_CTA + tile_step + 1
                        next_stage_idx = (tile_step + 1) % STAGE_COUNT
                        sIN_tile_next = sIN_tile0
                        tma_mbar_ptr_next = tma_mbar_ptr0
                        if cutlass.const_expr(STAGE_COUNT > 1):
                            tma_mbar_ptr_next = tma_mbar_ptr0 + next_stage_idx
                        if cutlass.const_expr(STAGE_COUNT > 1):
                            if next_stage_idx == 1:
                                sIN_tile_next = sIN_tile1

                        gIN_tile_next = cute.local_tile(
                            tma_tensor_in, (TILE_M, TILE_K), (m_tile, k_tile_next)
                        )
                        self._issue_tma_load(
                            tma_atom_in,
                            gIN_tile_next,
                            sIN_tile_next,
                            tma_mbar_ptr_next,
                            warp_idx,
                        )

                if warp_idx >= 1 and warp_idx <= compute_warps:
                    cute.arch.mbarrier_wait(tma_mbar_ptr, tma_phase)
                    lane = tidx % 32
                    m_lane = (warp_idx - 1) * 32 + lane

                    for mm in cutlass.range_constexpr(M_ITERS_PER_LANE):
                        m_rel = m_lane + mm * M_THREADS
                        m_input = m0 + m_rel
                        if cutlass.const_expr(IS_FULL_K_TILES):
                            m = m_input
                            is_valid = True
                            if cutlass.const_expr(USE_GROUP_ALIGNMENT):
                                m_padded, is_valid = self._compute_padded_m_index(
                                    cutlass.Int32(m_input), group_end_offsets, group_alignment_size, num_groups
                                )
                                m = cutlass.Int64(m_padded)
                            if m_rel < TILE_M and is_valid:
                                # Buffer scales for vectorized store
                                scale_buffer = cute.make_rmem_tensor(
                                    (K_BLOCKS_PER_TILE,), cutlass.Uint8
                                )

                                for kb in cutlass.range_constexpr(K_BLOCKS_PER_TILE):
                                    k_base = kb * SCALE_DIM_K_VALUE
                                    vals_block = self._load_vals_block_full(
                                        sIN_tile,
                                        m_rel,
                                        k_base,
                                    )

                                    amax = self._compute_amax(vals_block)

                                    scale_biased, inv_scale = (
                                        self._compute_scale_from_amax(amax, USE_RCEIL)
                                    )
                                    scale_buffer[kb] = cutlass.Uint8(scale_biased)

                                    self._quantize_store_full(
                                        vals_block,
                                        inv_scale,
                                        sOUT_tile,
                                        m_rel,
                                        k_base,
                                        USE_RCEIL,
                                    )

                                # Vectorized scale store
                                k_block_base = k_tile_eff * K_BLOCKS_PER_TILE
                                self._store_scales_vectorized(
                                    scales_tensor,
                                    m,
                                    k_block_base,
                                    scale_buffer,
                                    cutlass.Int32(K_BLOCKS_PER_TILE),
                                    BLOCKED_SCALE_OUTPUT_VALUE,
                                )
                        else:
                            m_in_bounds = m_input < M
                            m = m_input
                            is_valid = True
                            if cutlass.const_expr(USE_GROUP_ALIGNMENT):
                                m_padded, is_valid = self._compute_padded_m_index(
                                    cutlass.Int32(m_input), group_end_offsets, group_alignment_size, num_groups
                                )
                                m = cutlass.Int64(m_padded)
                            if m_rel < TILE_M and m_in_bounds and is_valid:
                                # Buffer scales for vectorized store
                                scale_buffer = cute.make_rmem_tensor(
                                    (K_BLOCKS_PER_TILE,), cutlass.Uint8
                                )
                                num_valid_scales = cutlass.Int32(0)

                                for kb in cutlass.range_constexpr(K_BLOCKS_PER_TILE):
                                    k_block = k_tile_eff * K_BLOCKS_PER_TILE + kb
                                    if k_block < k_blocks:
                                        k_base = kb * SCALE_DIM_K_VALUE
                                        vals_block = self._load_vals_block_tail(
                                            sIN_tile,
                                            k0,
                                            m_rel,
                                            k_base,
                                            K,
                                        )

                                        amax = self._compute_amax(vals_block)

                                        scale_biased, inv_scale = (
                                            self._compute_scale_from_amax(
                                                amax, USE_RCEIL
                                            )
                                        )
                                        scale_buffer[num_valid_scales] = cutlass.Uint8(
                                            scale_biased
                                        )
                                        num_valid_scales = num_valid_scales + 1

                                        self._quantize_store_tail(
                                            vals_block,
                                            inv_scale,
                                            sOUT_tile,
                                            k0,
                                            m_rel,
                                            k_base,
                                            K,
                                            USE_RCEIL,
                                        )

                                # Vectorized scale store
                                if num_valid_scales > 0:
                                    k_block_base = k_tile_eff * K_BLOCKS_PER_TILE
                                    self._store_scales_vectorized(
                                        scales_tensor,
                                        m,
                                        k_block_base,
                                        scale_buffer,
                                        num_valid_scales,
                                        BLOCKED_SCALE_OUTPUT_VALUE,
                                    )

                # Store output: use regular stores with padding when group alignment is enabled
                if cutlass.const_expr(USE_GROUP_ALIGNMENT):
                    self._store_output_with_padding(
                        sOUT_tile,
                        out_mk,
                        m0,
                        k0,
                        group_end_offsets,
                        group_alignment_size,
                        num_groups,
                        warp_idx,
                    )
                else:
                    gOUT_tile = cute.local_tile(
                        tma_tensor_out, (TILE_M, TILE_K), (m_tile, k_tile_eff)
                    )
                    self._issue_tma_store(
                        tma_atom_out,
                        gOUT_tile,
                        sOUT_tile,
                        warp_idx,
                    )

        @cute.jit
        def __call__(
            self,
            inp_mk: cute.Tensor,
            out_mk: cute.Tensor,
            scales_colwise_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            group_end_offsets: cute.Tensor,
            group_alignment_size: cutlass.Int32,
            num_groups: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(
                cute, TILE_M, TILE_K
            )
            # SM >= 100 (Blackwell and beyond, including consumer SM12x and
            # SM13x): use tcgen05.CtaGroup.ONE for the optimised single-CTA
            # Blackwell TMA load path.
            if cutlass.const_expr(IS_BLACKWELL_VALUE):
                g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            else:
                g2s_op = cpasync.CopyBulkTensorTileG2SOp()
            tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
                g2s_op,
                inp_mk,
                smem_layout_in,
                (TILE_M, TILE_K),
            )
            tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_mk,
                smem_layout_out,
                (TILE_M, TILE_K),
            )

            blocked_scale_layout = cute.make_layout((1,))
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                padded_scale_cols = cute.round_up(k_blocks, 4)
                m_block_tiles = cute.ceil_div(M, 128)
                k_block_tiles = padded_scale_cols // cutlass.Int64(4)
                blocked_scale_layout = cute.make_layout(
                    ((32, 4, m_block_tiles), (4, k_block_tiles)),
                    stride=(
                        (16, 4, cutlass.Int64(128) * padded_scale_cols),
                        (1, cutlass.Int64(512)),
                    ),
                )

            self.kernel(
                inp_mk,
                tma_atom_in,
                tma_tensor_in,
                out_mk,
                tma_atom_out,
                tma_tensor_out,
                scales_colwise_u8,
                M,
                K,
                k_blocks,
                m_cta_tiles,
                k_cta_tiles,
                blocked_scale_layout,
                group_end_offsets,
                group_alignment_size,
                num_groups,
                SCALE_DIM_K=SCALE_DIM_K_VALUE,
                USE_RCEIL=(scaling_mode == "rceil"),
                IS_FULL_K_TILES=IS_FULL_K_TILES_VALUE,
                STAGE_COUNT=STAGE_COUNT_VALUE,
                IS_BLACKWELL=IS_BLACKWELL_VALUE,
                USE_GROUP_ALIGNMENT=USE_GROUP_ALIGNMENT_VALUE,
            ).launch(
                grid=(k_cta_tiles, m_cta_tiles, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    kernel = Mxfp8Quantize2dKernel()

    m = cute.sym_int(divisibility=32)
    k = cute.sym_int(divisibility=32)
    kb = cute.sym_int()
    inp_stride0 = cute.sym_int()
    inp_stride1 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()
    scale_stride0 = cute.sym_int()
    scale_stride1 = cute.sym_int()

    # When using group alignment, output M dimension can be larger than input M
    if use_group_alignment:
        m_out = cute.sym_int(divisibility=32)
    else:
        m_out = m

    fake_inp = make_fake_tensor(
        INPUT_CUTLASS_DTYPE,
        (m, k),
        stride=(inp_stride0, inp_stride1),
    )
    fake_out = make_fake_tensor(
        cutlass.Float8E4M3FN,
        (m_out, k),
        stride=(out_stride0, out_stride1),
    )
    if blocked_scale_output:
        scale_flat = cute.sym_int()
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (scale_flat,),
            stride=(scale_stride0,),
        )
    else:
        # Scales correspond to output M dimension when using group alignment
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (m_out, kb),
            stride=(scale_stride0, scale_stride1),
        )

    # Create fake tensors for group alignment parameters
    if use_group_alignment:
        num_groups_sym = cute.sym_int()
        group_stride = cute.sym_int()
        fake_group_end_offsets = make_fake_tensor(
            cutlass.Int32,
            (num_groups_sym,),
            stride=(group_stride,),
        )
    else:
        # Create a dummy tensor with size 1 when not using group alignment
        fake_group_end_offsets = make_fake_tensor(
            cutlass.Int32,
            (1,),
            stride=(1,),
        )

    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        inp_mk=fake_inp,
        out_mk=fake_out,
        scales_colwise_u8=fake_scales,
        M=0,
        K=0,
        k_blocks=0,
        m_cta_tiles=1,
        k_cta_tiles=1,
        group_end_offsets=fake_group_end_offsets,
        group_alignment_size=1,
        num_groups=0,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def mxfp8_quantize_cutedsl_2d(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "floor",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    group_end_offsets: torch.Tensor | None = None,
    group_alignment_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to MXFP8 format using CuTe DSL kernel.

    Quantizes along the K dimension - each row has K//32 scales, one per block of 32 K elements.

    Args:
        x: Input tensor of shape (M, K) - unpadded
        block_size: Block size for quantization along K (only 32 supported)
        scaling_mode: Scaling mode ("floor" or "rceil")
        stage_count: Number of pipeline stages (1 or 2)
        blocked_scale_output: Whether to output scales in blocked layout
        group_end_offsets: Optional int32 tensor with end indexes of groups along M dimension.
            When provided, output will be padded to align each group to group_alignment_size.
        group_alignment_size: Alignment size for padding group sizes (default: 1, no padding).
            Each group will be padded to the next multiple of this value.

    Returns:
        q_data: Quantized data in row-major layout with shape:
            - (M, K) if group_end_offsets is None
            - (M + num_groups * group_alignment_size, K) if group_end_offsets is provided (overallocated)
        scales: Scales tensor with shape corresponding to q_data dimensions, in row-major or blocked layout

    Note:
        When group_end_offsets is provided, the output is overallocated to handle worst-case padding
        where each group needs group_alignment_size padding. The actual padded size per group depends
        on the group sizes. Padding rows contain uninitialized data and should be ignored based on
        the computed padded group offsets.
    """
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    M, K = x.shape
    assert K % block_size == 0, "K must be divisible by block_size"

    # Validate group alignment parameters
    use_group_alignment = group_end_offsets is not None
    if use_group_alignment:
        assert group_end_offsets.dtype == torch.int32, "group_end_offsets must be int32"
        assert group_end_offsets.is_cuda, "group_end_offsets must be CUDA"
        assert group_end_offsets.dim() == 1, "group_end_offsets must be 1D"
        assert group_alignment_size > 0, "group_alignment_size must be positive"

    # Calculate output M dimension with padding if using group alignment
    M_output = M
    if use_group_alignment:
        num_groups = len(group_end_offsets)
        # Overallocate to handle worst case: each group needs alignment_size padding
        M_output = M + num_groups * group_alignment_size

    _, config = _select_cutedsl_config(str(x.dtype), scaling_mode, K)
    compute_warps, tile_m, tile_k, k_tiles_per_cta = config
    
    assert stage_count >= 1, "stage_count must be >= 1"
    assert stage_count <= 2, "stage_count must be <= 2"
    is_full_k_tiles = K % (tile_k * k_tiles_per_cta) == 0
    is_sm_10x = torch.cuda.get_device_capability()[0] == 10
    if blocked_scale_output and not is_sm_10x:
        raise NotImplementedError(
            "blocked_scale_output is only supported on SM 10.x GPUs "
            "because it produces the tcgen05 blocked scale layout"
        )

    # Output in row-major layout: stride (K, 1).
    q_data = torch.empty_strided(
        (M_output, K),
        (K, 1),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    k_blocks = K // block_size
    if blocked_scale_output:
        padded_scale_rows = ceil_div(M_output, 128) * 128
        padded_scale_cols = ceil_div(k_blocks, 4) * 4
        scales_u8 = torch.empty(
            (padded_scale_rows * padded_scale_cols,),
            device=x.device,
            dtype=torch.uint8,
        )
    else:
        scales_u8 = torch.empty(
            (M_output, k_blocks),
            device=x.device,
            dtype=torch.uint8,
        )

    compiled = _compile_mxfp8_quantize_2d_cutedsl(
        str(x.dtype),
        scaling_mode,
        compute_warps,
        tile_m,
        tile_k,
        stage_count,
        k_tiles_per_cta,
        is_full_k_tiles,
        is_sm_10x,
        blocked_scale_output,
        use_group_alignment,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    # Note: Grid is based on input M (not padded M_output)
    # The kernel will map input M to output M with padding
    m_cta_tiles = ceil_div(M, tile_m)
    k_cta_tiles = ceil_div(K, tile_k * k_tiles_per_cta)

    # Prepare group alignment parameters
    if use_group_alignment:
        num_groups = len(group_end_offsets)
    else:
        # Create dummy tensor when not using group alignment
        group_end_offsets = torch.zeros(1, dtype=torch.int32, device=x.device)
        num_groups = 0

    compiled(
        x,
        q_data,
        scales_u8,
        int(M),  # Input M (unpadded)
        int(K),
        int(k_blocks),
        int(m_cta_tiles),
        int(k_cta_tiles),
        group_end_offsets,
        int(group_alignment_size),
        int(num_groups),
        stream,
    )

    return q_data, scales_u8.view(torch.float8_e8m0fnu)
