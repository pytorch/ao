# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Optional

import torch


def _float8_name(dtype: torch.dtype) -> str:
    if dtype == torch.float8_e4m3fn:
        return "e4m3"
    if dtype == torch.float8_e5m2:
        return "e5m2"
    raise AssertionError(f"Unsupported FP8 dtype: {dtype}")


def _float_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise AssertionError(f"Unsupported dtype: {dtype}")


def _sparse_wgmma_opcode(input_dtype: str, weight_dtype: str, n: int) -> str:
    if n not in (32, 64, 128):
        raise AssertionError(f"Unsupported sparse WGMMA N: {n}")
    if input_dtype not in ("e4m3", "e5m2") or weight_dtype not in ("e4m3", "e5m2"):
        raise AssertionError(
            f"Unsupported sparse WGMMA dtypes: {weight_dtype}, {input_dtype}"
        )
    return (
        f"wgmma.mma_async.sp.sync.aligned.m64n{n}k64.f32.{weight_dtype}.{input_dtype}"
    )


_WGMMA_WEIGHT_ROWS = 64
_WGMMA_INPUT_ROWS = 128
_WGMMA_LOGICAL_K = 64
_WGMMA_COMPRESSED_K = _WGMMA_LOGICAL_K // 2
_WGMMA_THREADS = 128
_WGMMA_ACC_REGS = _WGMMA_WEIGHT_ROWS * _WGMMA_INPUT_ROWS // _WGMMA_THREADS
_WARPS_PER_WARPGROUP = 4
_MMA_PER_WG = 2
_CONSUMER_WARPGROUPS = 2
_PRODUCER_WARPGROUPS = 1
_WARPGROUPS = _CONSUMER_WARPGROUPS + _PRODUCER_WARPGROUPS
_CTA_THREADS = _WGMMA_THREADS * _WARPGROUPS
_WG_WEIGHT_ROWS = _WGMMA_WEIGHT_ROWS * _MMA_PER_WG
_CTA_WEIGHT_ROWS = _WG_WEIGHT_ROWS * _CONSUMER_WARPGROUPS
_TOTAL_ACC_REGS = _WGMMA_ACC_REGS * _MMA_PER_WG
_PRODUCER_WARP = _CONSUMER_WARPGROUPS * _WARPS_PER_WARPGROUP
_PRODUCER_REGS = 32
_CONSUMER_REGS = 232
_A_TILE_BYTES = _WGMMA_WEIGHT_ROWS * _WGMMA_COMPRESSED_K
_STAGES = 4
_K_GROUP = 2
_BUFFERS = _STAGES * _K_GROUP
_A_STAGE_BYTES = _CTA_WEIGHT_ROWS * _WGMMA_COMPRESSED_K
_B_STAGE_BYTES = _WGMMA_INPUT_ROWS * _WGMMA_LOGICAL_K
_GMMA_LAYOUT_TYPE_B64 = 2
_GMMA_LAYOUT_TYPE_B32 = 3
_GMMA_DESC_UNIT_BYTES = 16
_GMMA_DESC_ROWS_PER_GROUP = 8
_GMMA_DESC_LEADING_OFFSET = 1
_A_GMMA_DESC_STRIDE_OFFSET = (
    _GMMA_DESC_ROWS_PER_GROUP * _WGMMA_COMPRESSED_K // _GMMA_DESC_UNIT_BYTES
)
_B_GMMA_DESC_STRIDE_OFFSET = (
    _GMMA_DESC_ROWS_PER_GROUP * _WGMMA_LOGICAL_K // _GMMA_DESC_UNIT_BYTES
)
_TMA_STAGE_BYTES = _A_STAGE_BYTES + _B_STAGE_BYTES
_TMA_GROUP_BYTES = _K_GROUP * _TMA_STAGE_BYTES
_A_STAGE_DESC_UNITS = _A_STAGE_BYTES // _GMMA_DESC_UNIT_BYTES
_B_STAGE_DESC_UNITS = _B_STAGE_BYTES // _GMMA_DESC_UNIT_BYTES

_META_UINT32_PER_ROW = 4
_META_HALVES_PER_GROUP = max(1, _K_GROUP // 2)
_META_STAGE_UINT32 = _CTA_WEIGHT_ROWS * _META_UINT32_PER_ROW
_META_STAGE_BYTES = _META_STAGE_UINT32 * 4
_META_GROUP_BYTES = _META_HALVES_PER_GROUP * _META_STAGE_BYTES
_META_SMEM_UINT32 = _STAGES * _META_HALVES_PER_GROUP * _META_STAGE_UINT32
_RASTER_GROUP_N = 8
_META_SMEM_STAGED = _K_GROUP % 2 == 0
_EPI_TILE_M = 64
_EPI_TILE_N = 64
_EPI_STAGES = 2
_EPI_M_SUBTILES = _WGMMA_INPUT_ROWS // _EPI_TILE_M
_EPI_N_SUBTILES = _WG_WEIGHT_ROWS // _EPI_TILE_N
_EPI_SLOTS = _CONSUMER_WARPGROUPS * _EPI_STAGES
_EPI_VID2_PER_SUBTILE = _WGMMA_ACC_REGS // 4 // _EPI_M_SUBTILES


@functools.cache
def _compile_rowwise_scaled_linear_sparse_cutedsl(
    input_dtype: str,
    weight_dtype: str,
    scale_dtype: str,
    output_dtype: str,
    has_bias: bool,
    m_exact: bool,
    n_exact: bool,
    cluster_mcast: bool,
    meta_block_safe: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.cutlass_dsl import T, dsl_user_op

    if scale_dtype not in ("fp16", "bf16", "fp32"):
        raise AssertionError(f"Unsupported scale dtype: {scale_dtype}")
    if output_dtype not in ("fp16", "bf16"):
        raise AssertionError(f"Unsupported output dtype: {output_dtype}")
    opcode = _sparse_wgmma_opcode(input_dtype, weight_dtype, _WGMMA_INPUT_ROWS)
    _use_a_mcast = cluster_mcast
    _meta_staged = _META_SMEM_STAGED and meta_block_safe

    INPUT_DTYPE = cutlass.Float8E5M2 if input_dtype == "e5m2" else cutlass.Float8E4M3FN
    WEIGHT_DTYPE = (
        cutlass.Float8E5M2 if weight_dtype == "e5m2" else cutlass.Float8E4M3FN
    )
    if scale_dtype == "fp16":
        SCALE_DTYPE = cutlass.Float16
    elif scale_dtype == "bf16":
        SCALE_DTYPE = cutlass.BFloat16
    else:
        SCALE_DTYPE = cutlass.Float32
    OUTPUT_DTYPE = cutlass.Float16 if output_dtype == "fp16" else cutlass.BFloat16

    @dsl_user_op
    def _pack_gmma_smem_desc_k(
        ptr: cute.Pointer,
        byte_offset,
        leading: int,
        stride: int,
        layout_type: int,
        *,
        loc=None,
        ip=None,
    ):
        address = ptr.toint(loc=loc, ip=ip).to(cutlass.Uint64) + cutlass.Uint64(
            byte_offset
        ).to(cutlass.Uint64)
        start = (address >> cutlass.Uint64(4)) & cutlass.Uint64(0x3FFF)
        leading_bits = cutlass.Uint64(leading & 0x3FFF) << cutlass.Uint64(16)
        stride_bits = cutlass.Uint64(stride & 0x3FFF) << cutlass.Uint64(32)
        layout_bits = cutlass.Uint64(layout_type) << cutlass.Uint64(62)
        return (start | leading_bits | stride_bits | layout_bits).to(cutlass.Int64)

    @dsl_user_op
    def _wgmma_sp(
        desc_a,
        desc_b,
        meta: cutlass.Uint32,
        acc,
        *,
        loc=None,
        ip=None,
    ):
        regs = ", ".join(f"${i}" for i in range(_WGMMA_ACC_REGS))
        desc_a_arg = 2 * _WGMMA_ACC_REGS
        asm = (
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            f"{opcode} "
            "{" + regs + "}, "
            f"${desc_a_arg}, ${desc_a_arg + 1}, ${desc_a_arg + 2}, 0, p, 1, 1;\n"
            "}\n"
        )
        constraints = ",".join(
            ["=f"] * _WGMMA_ACC_REGS
            + [str(i) for i in range(_WGMMA_ACC_REGS)]
            + ["l", "l", "r"]
        )
        result = llvm.inline_asm(
            ir.Type.parse(
                "!llvm.struct<(" + ",".join(["f32"] * _WGMMA_ACC_REGS) + ")>"
            ),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip) for a in acc]
            + [
                cutlass.Int64(desc_a).ir_value(loc=loc, ip=ip),
                cutlass.Int64(desc_b).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(meta).ir_value(loc=loc, ip=ip),
            ],
            asm,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return tuple(
            cutlass.Float32(llvm.extractvalue(T.f32(), result, [i]))
            for i in range(_WGMMA_ACC_REGS)
        )

    def _acc_only_asm(body, n_acc):
        @dsl_user_op
        def op(acc, *, loc=None, ip=None):
            constraints = ",".join(["=f"] * n_acc + [str(i) for i in range(n_acc)])
            result = llvm.inline_asm(
                ir.Type.parse("!llvm.struct<(" + ",".join(["f32"] * n_acc) + ")>"),
                [cutlass.Float32(a).ir_value(loc=loc, ip=ip) for a in acc],
                "{\n" + body + "}\n",
                constraints,
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
            return tuple(
                cutlass.Float32(llvm.extractvalue(T.f32(), result, [i]))
                for i in range(n_acc)
            )

        return op

    _wgmma_fence = _acc_only_asm("wgmma.fence.sync.aligned;\n", _TOTAL_ACC_REGS)
    _wgmma_commit_wait = _acc_only_asm(
        "wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 1;\n",
        _TOTAL_ACC_REGS,
    )
    _wgmma_drain = _acc_only_asm(
        "wgmma.wait_group.sync.aligned 0;\n",
        _TOTAL_ACC_REGS,
    )

    @cute.struct
    class SharedStorage:
        sa: cute.struct.Align[
            cute.struct.MemRange[WEIGHT_DTYPE, _BUFFERS * _A_STAGE_BYTES],
            1024,
        ]
        sb: cute.struct.Align[
            cute.struct.MemRange[INPUT_DTYPE, _BUFFERS * _B_STAGE_BYTES],
            1024,
        ]
        smeta: cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint32, _META_SMEM_UINT32],
            128,
        ]
        sepi: cute.struct.Align[
            cute.struct.MemRange[OUTPUT_DTYPE, _EPI_SLOTS * _EPI_TILE_M * _EPI_TILE_N],
            1024,
        ]
        mbar: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, _STAGES],
            8,
        ]
        mbar_empty: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, _STAGES],
            8,
        ]

    def _make_smem_layouts():
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,
            WEIGHT_DTYPE,
        )
        b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64,
            INPUT_DTYPE,
        )
        a_staged = cute.tile_to_shape(
            a_atom,
            (_CTA_WEIGHT_ROWS, _WGMMA_COMPRESSED_K, _BUFFERS),
            order=(0, 1, 2),
        )
        b_staged = cute.tile_to_shape(
            b_atom,
            (_WGMMA_INPUT_ROWS, _WGMMA_LOGICAL_K, _BUFFERS),
            order=(0, 1, 2),
        )
        return a_staged, b_staged

    def _make_epi_smem_layout():
        c_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            OUTPUT_DTYPE,
        )
        return cute.tile_to_shape(
            c_atom,
            (_EPI_TILE_M, _EPI_TILE_N),
            order=(0, 1),
        )

    class RowwiseScaledLinearSparseWgmma:
        def __init__(self):
            self.shared_storage = SharedStorage

        @cute.kernel
        def kernel(
            self,
            tma_atom_a: cute.CopyAtom,
            tma_tensor_a: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            tma_tensor_b: cute.Tensor,
            tma_atom_c: cute.CopyAtom,
            tma_tensor_c: cute.Tensor,
            input_scale: cute.Tensor,
            weight_meta: cute.Tensor,
            weight_scale: cute.Tensor,
            bias: cute.Tensor,
            output: cute.Tensor,
            m_size: cutlass.Int32,
            n_size: cutlass.Int32,
            k_size: cutlass.Int32,
            metadata_rows: cutlass.Int32,
            raster_group_n: cutlass.Int32,
        ):
            import cutlass.utils as utils

            tidx, _, _ = cute.arch.thread_idx()
            raw_x, raw_y, _ = cute.arch.block_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            warpgroup = tidx // _WGMMA_THREADS
            tid_wg = tidx - warpgroup * _WGMMA_THREADS
            a_mcast_mask = cutlass.Int16(0)
            a_tma_crd = cutlass.Int32(0)
            a_tma_layout = cute.make_layout(1)
            cta_rank = cutlass.Int32(0)
            peer_cta_rank = cutlass.Int32(0)
            if cutlass.const_expr(cluster_mcast):
                cta_layout_mnk = cute.make_layout((1, 2, 1))
                cta_layout_vmnk = cute.make_layout((1, 1, 2, 1))
                cidx, cidy, _ = cute.arch.cluster_idx()
                raw_x = cidx
                raw_y = cidy
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank)
                cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
                peer_cta_rank = cta_rank ^ 1
                if cutlass.const_expr(_use_a_mcast):
                    a_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask(
                        cta_layout_vmnk, cluster_coord_vmnk, 2
                    )
                    a_tma_layout = cute.make_layout(
                        cute.slice_(cta_layout_mnk, (0, None, 0)).shape
                    )
                    a_tma_crd = cluster_coord_mnk[1]

            num_n_blocks = (n_size + _CTA_WEIGHT_ROWS - 1) // _CTA_WEIGHT_ROWS
            num_m_blocks = (m_size + _WGMMA_INPUT_ROWS - 1) // _WGMMA_INPUT_ROWS
            if cutlass.const_expr(cluster_mcast):
                num_m_units = num_m_blocks // 2
            else:
                num_m_units = num_m_blocks
            linear_tile = raw_y * num_n_blocks + raw_x
            tiles_per_group = raster_group_n * num_m_units
            raster_group = linear_tile // tiles_per_group
            first_n = raster_group * raster_group_n
            group_n = num_n_blocks - first_n
            if group_n > raster_group_n:
                group_n = raster_group_n
            tile_in_group = linear_tile - raster_group * tiles_per_group
            n_block = first_n + tile_in_group % group_n
            m_unit = tile_in_group // group_n
            if cutlass.const_expr(cluster_mcast):
                m_block = m_unit * 2 + cta_rank
            else:
                m_block = m_unit

            n_base = n_block * _CTA_WEIGHT_ROWS
            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(self.shared_storage)
            a_staged, b_staged = _make_smem_layouts()
            sa = storage.sa.get_tensor(a_staged)
            sb = storage.sb.get_tensor(b_staged)
            smeta = storage.smeta.get_tensor(cute.make_layout(_META_SMEM_UINT32))
            epi_layout = _make_epi_smem_layout()
            sepi_ptr = storage.sepi.data_ptr()
            mbar = storage.mbar.data_ptr()
            mbar_empty = storage.mbar_empty.data_ptr()
            bulk_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyBulkG2SOp(),
                cutlass.Uint32,
            )

            if warp_idx == 0:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(_STAGES):
                        cute.arch.mbarrier_init(mbar + stage, 1)
                        empty_arrivals = _CONSUMER_WARPGROUPS
                        if cutlass.const_expr(cluster_mcast):
                            empty_arrivals = _CONSUMER_WARPGROUPS * 2
                        cute.arch.mbarrier_init(mbar_empty + stage, empty_arrivals)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(cluster_mcast):
                cute.arch.cluster_arrive_relaxed()

            meta_stage_base = n_base
            if cutlass.const_expr(_meta_staged and not n_exact):
                meta_stage_base = cutlass.min(n_base, metadata_rows - _CTA_WEIGHT_ROWS)
            meta_shift = n_base - meta_stage_base

            k_tiles = (k_size + _WGMMA_LOGICAL_K - 1) // _WGMMA_LOGICAL_K
            meta_last_half = (k_tiles - 1) // 2
            num_groups = (k_tiles + _K_GROUP - 1) // _K_GROUP
            num_waves = (num_groups + _STAGES - 1) // _STAGES
            groups_rounded = num_waves * _STAGES
            meta_row_stride = metadata_rows * 4
            meta_u32 = cute.recast_ptr(weight_meta.iterator, dtype=cutlass.Uint32)

            def issue_group(
                group,
                stage,
                stage_const,
                bar,
                atom_a,
                ga,
                sa_,
                atom_b,
                gb,
                sb_,
                atom_m,
                msrc,
                mstride,
                smeta_,
                last_half,
                staged,
                a_mask,
                use_a_mcast,
                mma_per_wg,
                tma_group_bytes,
                meta_stage_uint32,
                meta_group_bytes,
            ):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar + stage,
                        tma_group_bytes + (meta_group_bytes if staged else 0),
                    )
                for sub in cutlass.range_constexpr(_K_GROUP):
                    buf = stage * _K_GROUP + sub
                    kt = group * _K_GROUP + sub
                    a_dst = sa_[stage_const * _K_GROUP + sub]
                    if cutlass.const_expr(use_a_mcast):
                        cute.copy(
                            atom_a.with_(
                                mcast_mask=a_mask,
                                tma_bar_ptr=bar + stage,
                            ),
                            ga[None, kt],
                            a_dst,
                        )
                    else:
                        cute.copy(
                            atom_a,
                            ga[None, kt],
                            a_dst,
                            tma_bar_ptr=bar + stage,
                        )
                    cute.copy(
                        atom_b,
                        gb[None, kt],
                        sb_[None, buf],
                        tma_bar_ptr=bar + stage,
                    )
                if cutlass.const_expr(staged):
                    for i in cutlass.range_constexpr(_META_HALVES_PER_GROUP):
                        half = group * _META_HALVES_PER_GROUP + i
                        if half > last_half:
                            half = last_half
                        slot = stage * _META_HALVES_PER_GROUP + i
                        cute.copy(
                            atom_m,
                            cute.make_tensor(
                                msrc + half * mstride,
                                cute.make_layout(meta_stage_uint32),
                            ),
                            cute.make_tensor(
                                smeta_.iterator + slot * meta_stage_uint32,
                                cute.make_layout(meta_stage_uint32),
                            ),
                            mbar_ptr=bar + stage,
                        )

            def group_metas_smem(stage, smeta_, lanes, mma_per_wg, meta_stage_uint32):
                out = ()
                for sub in cutlass.range_constexpr(_K_GROUP):
                    slot = stage * _META_HALVES_PER_GROUP + sub // 2
                    for j in cutlass.range_constexpr(mma_per_wg):
                        out = out + (
                            smeta_[slot * meta_stage_uint32 + lanes[j] + 2 * (sub % 2)],
                        )
                return out

            def group_metas(group, bases, row_strides, k_tiles_, mma_per_wg):
                out = ()
                for sub in cutlass.range_constexpr(_K_GROUP):
                    kt = group * _K_GROUP + sub
                    if kt >= k_tiles_:
                        kt = k_tiles_ - 1
                    half = kt // 2
                    for j in cutlass.range_constexpr(mma_per_wg):
                        out = out + (
                            (
                                bases[j] + (half * row_strides[j] + 2 * (kt - half * 2))
                            ).load(),
                        )
                return out

            if cutlass.const_expr(cluster_mcast):
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()

            if warpgroup == _CONSUMER_WARPGROUPS:
                cute.arch.setmaxregister_decrease(_PRODUCER_REGS)
                if warp_idx == _PRODUCER_WARP:
                    ga = cute.local_tile(
                        tma_tensor_a,
                        (_CTA_WEIGHT_ROWS, _WGMMA_COMPRESSED_K),
                        (n_block, None),
                    )
                    gb = cute.local_tile(
                        tma_tensor_b,
                        (_WGMMA_INPUT_ROWS, _WGMMA_LOGICAL_K),
                        (m_block, None),
                    )
                    _, tma_ga = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_a,
                        a_tma_crd,
                        a_tma_layout,
                        cute.group_modes(sa, 0, 2),
                        cute.group_modes(ga, 0, 2),
                    )
                    tma_sa = ()
                    for b in cutlass.range_constexpr(_BUFFERS):
                        sa_buf, _ = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_a,
                            a_tma_crd,
                            a_tma_layout,
                            cute.group_modes(cute.slice_(sa, (None, None, b)), 0, 2),
                            cute.group_modes(ga, 0, 2),
                        )
                        tma_sa = tma_sa + (sa_buf,)
                    tma_sb, tma_gb = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_b,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sb, 0, 2),
                        cute.group_modes(gb, 0, 2),
                    )
                    meta_src = meta_u32 + meta_stage_base * _META_UINT32_PER_ROW
                    for st in cutlass.range_constexpr(_STAGES):
                        issue_group(
                            cutlass.Int32(st),
                            cutlass.Int32(st),
                            st,
                            mbar,
                            tma_atom_a,
                            tma_ga,
                            tma_sa,
                            tma_atom_b,
                            tma_gb,
                            tma_sb,
                            bulk_atom,
                            meta_src,
                            meta_row_stride,
                            smeta,
                            meta_last_half,
                            _meta_staged,
                            a_mcast_mask,
                            _use_a_mcast,
                            _MMA_PER_WG,
                            _TMA_GROUP_BYTES,
                            _META_STAGE_UINT32,
                            _META_GROUP_BYTES,
                        )
                    wave = cutlass.Int32(0)
                    while wave < num_waves:
                        phase = wave & 1
                        base_group = wave * _STAGES
                        for st in cutlass.range_constexpr(_STAGES):
                            prefetch_group = base_group + st + _STAGES
                            if prefetch_group < groups_rounded:
                                cute.arch.mbarrier_wait(mbar_empty + st, phase)
                                issue_group(
                                    prefetch_group,
                                    cutlass.Int32(st),
                                    st,
                                    mbar,
                                    tma_atom_a,
                                    tma_ga,
                                    tma_sa,
                                    tma_atom_b,
                                    tma_gb,
                                    tma_sb,
                                    bulk_atom,
                                    meta_src,
                                    meta_row_stride,
                                    smeta,
                                    meta_last_half,
                                    _meta_staged,
                                    a_mcast_mask,
                                    _use_a_mcast,
                                    _MMA_PER_WG,
                                    _TMA_GROUP_BYTES,
                                    _META_STAGE_UINT32,
                                    _META_GROUP_BYTES,
                                )
                        wave = wave + 1
            else:
                cute.arch.setmaxregister_increase(_CONSUMER_REGS)
                wg_base = n_base + warpgroup * _WG_WEIGHT_ROWS
                m_base = m_block * _WGMMA_INPUT_ROWS
                desc_a_bases = tuple(
                    _pack_gmma_smem_desc_k(
                        sa.iterator,
                        (warpgroup * _MMA_PER_WG + j) * _A_TILE_BYTES,
                        _GMMA_DESC_LEADING_OFFSET,
                        _A_GMMA_DESC_STRIDE_OFFSET,
                        _GMMA_LAYOUT_TYPE_B32,
                    )
                    for j in range(_MMA_PER_WG)
                )
                desc_b_base = _pack_gmma_smem_desc_k(
                    sb.iterator,
                    0,
                    _GMMA_DESC_LEADING_OFFSET,
                    _B_GMMA_DESC_STRIDE_OFFSET,
                    _GMMA_LAYOUT_TYPE_B64,
                )
                meta_lane_row = (
                    (tid_wg // 4) % 8 + 8 * (tid_wg % 2) + 16 * (tid_wg // 32)
                )
                meta_k_half = (tid_wg // 2) % 2
                if cutlass.const_expr(_meta_staged):
                    meta_smem_lanes = ()
                    for j in cutlass.range_constexpr(_MMA_PER_WG):
                        slot_row = (
                            (warpgroup * _MMA_PER_WG + j) * _WGMMA_WEIGHT_ROWS
                            + meta_lane_row
                            + meta_shift
                        )
                        if cutlass.const_expr(not n_exact):
                            if slot_row >= _CTA_WEIGHT_ROWS:
                                slot_row = _CTA_WEIGHT_ROWS - 1
                        meta_smem_lanes = meta_smem_lanes + (
                            slot_row * _META_UINT32_PER_ROW + meta_k_half,
                        )
                else:
                    meta_ptrs = ()
                    meta_strides = ()
                    for j in cutlass.range_constexpr(_MMA_PER_WG):
                        meta_row = wg_base + j * _WGMMA_WEIGHT_ROWS + meta_lane_row
                        meta_row_tile = meta_row // 64
                        meta_row_in_tile = meta_row - meta_row_tile * 64
                        meta_base = (
                            meta_row_tile * 256 + meta_row_in_tile * 4 + meta_k_half
                        )
                        meta_stride = meta_row_stride
                        if cutlass.const_expr(not n_exact):
                            if meta_row >= metadata_rows:
                                meta_base = cutlass.Int32(0)
                                meta_stride = cutlass.Int32(0)
                        meta_ptrs = meta_ptrs + (meta_u32 + meta_base,)
                        meta_strides = meta_strides + (meta_stride,)

                acc_col = m_base + 2 * (tid_wg % 4)
                w_scaled = ()
                w_bias = ()
                for j in cutlass.range_constexpr(_MMA_PER_WG):
                    acc_row = (
                        wg_base
                        + j * _WGMMA_WEIGHT_ROWS
                        + 16 * (tid_wg // 32)
                        + (tid_wg // 4) % 8
                    )
                    for vid1 in cutlass.range_constexpr(2):
                        weight_row = acc_row + 8 * vid1
                        row_safe = weight_row
                        if cutlass.const_expr(not n_exact):
                            if weight_row >= n_size:
                                row_safe = n_size - 1
                        w_scaled = w_scaled + (
                            (weight_scale.iterator + row_safe)
                            .load()
                            .to(cutlass.Float32),
                        )
                        if cutlass.const_expr(has_bias):
                            w_bias = w_bias + (
                                (bias.iterator + row_safe).load().to(cutlass.Float32),
                            )
                in_scaled = ()
                for vid2 in cutlass.range_constexpr(_WGMMA_ACC_REGS // 4):
                    for vid0 in cutlass.range_constexpr(2):
                        input_row = acc_col + vid0 + 8 * vid2
                        col_safe = input_row
                        if cutlass.const_expr(not m_exact):
                            if input_row >= m_size:
                                col_safe = m_size - 1
                        in_scaled = in_scaled + (
                            (input_scale.iterator + col_safe)
                            .load()
                            .to(cutlass.Float32),
                        )
                acc = tuple(cutlass.Float32(0.0) for _ in range(_TOTAL_ACC_REGS))
                metas = ()
                if cutlass.const_expr(not _meta_staged):
                    for st in cutlass.range_constexpr(_STAGES):
                        metas = metas + group_metas(
                            cutlass.Int32(st),
                            meta_ptrs,
                            meta_strides,
                            k_tiles,
                            _MMA_PER_WG,
                        )
                wave = cutlass.Int32(0)
                while wave < num_waves:
                    phase = wave & 1
                    base_group = wave * _STAGES
                    next_metas = ()
                    n_meta = _K_GROUP * _MMA_PER_WG
                    for st in cutlass.range_constexpr(_STAGES):
                        group = base_group + st
                        if cutlass.const_expr(_meta_staged):
                            cute.arch.mbarrier_wait(mbar + st, phase)
                            cute.arch.fence_view_async_shared()
                            cur_meta = group_metas_smem(
                                st,
                                smeta,
                                meta_smem_lanes,
                                _MMA_PER_WG,
                                _META_STAGE_UINT32,
                            )
                        else:
                            cur_meta = metas[st * n_meta : (st + 1) * n_meta]
                            next_metas = next_metas + group_metas(
                                group + _STAGES,
                                meta_ptrs,
                                meta_strides,
                                k_tiles,
                                _MMA_PER_WG,
                            )
                            cute.arch.mbarrier_wait(mbar + st, phase)
                            cute.arch.fence_view_async_shared()
                        acc = _wgmma_fence(acc)
                        for sub in cutlass.range_constexpr(_K_GROUP):
                            buf = st * _K_GROUP + sub
                            for j in cutlass.range_constexpr(_MMA_PER_WG):
                                lo = j * _WGMMA_ACC_REGS
                                hi = lo + _WGMMA_ACC_REGS
                                part = _wgmma_sp(
                                    desc_a_bases[j] + buf * _A_STAGE_DESC_UNITS,
                                    desc_b_base + buf * _B_STAGE_DESC_UNITS,
                                    cur_meta[sub * _MMA_PER_WG + j],
                                    acc[lo:hi],
                                )
                                acc = acc[:lo] + part + acc[hi:]
                        acc = _wgmma_commit_wait(acc)
                        prev_st = (st - 1) % _STAGES
                        released = cutlass.const_expr(st > 0) or wave > 0
                        if tid_wg == 0:
                            if released:
                                cute.arch.mbarrier_arrive(mbar_empty + prev_st)
                                if cutlass.const_expr(cluster_mcast):
                                    cute.arch.mbarrier_arrive(
                                        mbar_empty + prev_st,
                                        peer_cta_rank_in_cluster=peer_cta_rank,
                                    )
                    if cutlass.const_expr(not _meta_staged):
                        metas = next_metas
                    wave = wave + 1
                acc = _wgmma_drain(acc)

                epi_row = 2 * (tid_wg % 4)
                epi_col = 16 * (tid_wg // 32) + (tid_wg // 4) % 8
                epi_bar = 1 + warpgroup
                epi_warp = tid_wg // 32
                for s_i in cutlass.range_constexpr(_EPI_M_SUBTILES):
                    for t in cutlass.range_constexpr(_EPI_N_SUBTILES):
                        slot = warpgroup * _EPI_STAGES + (
                            (s_i * _EPI_N_SUBTILES + t) % _EPI_STAGES
                        )
                        sc = cute.make_tensor(
                            sepi_ptr + slot * (_EPI_TILE_M * _EPI_TILE_N),
                            epi_layout,
                        )
                        for v in cutlass.range_constexpr(_EPI_VID2_PER_SUBTILE):
                            vid2 = s_i * _EPI_VID2_PER_SUBTILE + v
                            for vid0 in cutlass.range_constexpr(2):
                                in_s = in_scaled[vid2 * 2 + vid0]
                                for vid1 in cutlass.range_constexpr(2):
                                    idx = t * 2 + vid1
                                    vid = vid0 + 2 * vid1 + 4 * vid2
                                    scaled = (
                                        acc[t * _WGMMA_ACC_REGS + vid]
                                        * w_scaled[idx]
                                        * in_s
                                    )
                                    if cutlass.const_expr(has_bias):
                                        scaled = scaled + w_bias[idx]
                                    sc[
                                        epi_row + (vid0 ^ (v & 1)) + 8 * v,
                                        epi_col + 8 * (vid1 ^ vid0),
                                    ] = scaled.to(OUTPUT_DTYPE)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(
                            barrier_id=epi_bar,
                            number_of_threads=_WGMMA_THREADS,
                        )
                        if epi_warp == 0:
                            gc = cute.local_tile(
                                tma_tensor_c,
                                (_EPI_TILE_M, _EPI_TILE_N),
                                (
                                    m_block * _EPI_M_SUBTILES + s_i,
                                    n_block * (_CTA_WEIGHT_ROWS // _EPI_TILE_N)
                                    + warpgroup * _EPI_N_SUBTILES
                                    + t,
                                ),
                            )
                            tma_sc, tma_gc = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_c,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sc, 0, 2),
                                cute.group_modes(gc, 0, 2),
                            )
                            cute.copy(tma_atom_c, tma_sc, tma_gc)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(
                                _EPI_STAGES - 1, read=True
                            )
                        cute.arch.barrier(
                            barrier_id=epi_bar,
                            number_of_threads=_WGMMA_THREADS,
                        )
                if epi_warp == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

        @cute.jit
        def __call__(
            self,
            input: cute.Tensor,
            input_scale: cute.Tensor,
            weight: cute.Tensor,
            weight_meta: cute.Tensor,
            weight_scale: cute.Tensor,
            bias: cute.Tensor,
            output: cute.Tensor,
            m_size: cutlass.Int32,
            n_size: cutlass.Int32,
            k_size: cutlass.Int32,
            metadata_rows: cutlass.Int32,
            num_n_blocks: cutlass.Int32,
            num_m_blocks: cutlass.Int32,
            raster_group_n: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            a_staged, b_staged = _make_smem_layouts()
            tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
                (
                    cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
                    if cutlass.const_expr(_use_a_mcast)
                    else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
                ),
                weight,
                cute.slice_(a_staged, (None, None, 0)),
                (_CTA_WEIGHT_ROWS, _WGMMA_COMPRESSED_K),
                num_multicast=2 if cutlass.const_expr(_use_a_mcast) else 1,
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                input,
                cute.slice_(b_staged, (None, None, 0)),
                (_WGMMA_INPUT_ROWS, _WGMMA_LOGICAL_K),
            )
            tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
                output,
                _make_epi_smem_layout(),
                (_EPI_TILE_M, _EPI_TILE_N),
            )
            if cutlass.const_expr(cluster_mcast):
                self.kernel(
                    tma_atom_a,
                    tma_tensor_a,
                    tma_atom_b,
                    tma_tensor_b,
                    tma_atom_c,
                    tma_tensor_c,
                    input_scale,
                    weight_meta,
                    weight_scale,
                    bias,
                    output,
                    m_size,
                    n_size,
                    k_size,
                    metadata_rows,
                    raster_group_n,
                ).launch(
                    grid=(num_n_blocks, num_m_blocks, 1),
                    block=[_CTA_THREADS, 1, 1],
                    cluster=(1, 2, 1),
                    smem=self.shared_storage.size_in_bytes(),
                    stream=stream,
                )
            else:
                self.kernel(
                    tma_atom_a,
                    tma_tensor_a,
                    tma_atom_b,
                    tma_tensor_b,
                    tma_atom_c,
                    tma_tensor_c,
                    input_scale,
                    weight_meta,
                    weight_scale,
                    bias,
                    output,
                    m_size,
                    n_size,
                    k_size,
                    metadata_rows,
                    raster_group_n,
                ).launch(
                    grid=(num_n_blocks, num_m_blocks, 1),
                    block=[_CTA_THREADS, 1, 1],
                    smem=self.shared_storage.size_in_bytes(),
                    stream=stream,
                )

    return cute.compile(
        RowwiseScaledLinearSparseWgmma(),
        input=make_fake_tensor(
            INPUT_DTYPE,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        input_scale=make_fake_tensor(
            SCALE_DTYPE,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        ),
        weight=make_fake_tensor(
            WEIGHT_DTYPE,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        weight_meta=make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        weight_scale=make_fake_tensor(
            SCALE_DTYPE,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        ),
        bias=make_fake_tensor(
            OUTPUT_DTYPE,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        ),
        output=make_fake_tensor(
            OUTPUT_DTYPE,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        m_size=_WGMMA_INPUT_ROWS,
        n_size=_CTA_WEIGHT_ROWS,
        k_size=_WGMMA_LOGICAL_K,
        metadata_rows=64,
        raster_group_n=1,
        num_n_blocks=1,
        num_m_blocks=1,
        stream=make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def rowwise_scaled_linear_sparse_cutedsl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_meta: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert input.dim() >= 2, f"Expected input to be 2D or higher, got {input.dim()}D"
    assert input.is_cuda, "Expected input to be CUDA"
    assert input.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "Expected FP8 input"
    assert weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
        "Expected FP8 weight"
    )
    assert input_scale.dtype == weight_scale.dtype, "Scale dtypes must match"
    assert input_scale.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        "Expected fp16, bf16, or fp32 scales"
    )
    output_dtype = out_dtype if out_dtype is not None else input_scale.dtype
    assert output_dtype in (torch.float16, torch.bfloat16), (
        "Expected fp16 or bf16 output"
    )
    if bias is not None:
        assert bias.dtype == output_dtype, "Bias dtype must match output dtype"
        assert bias.dim() == 1, "Expected bias to be 1D"

    k = input.shape[-1]
    assert k % 32 == 0, "Input K must be divisible by 32"
    assert weight.dim() == 2, f"Expected weight to be 2D, got {weight.dim()}D"
    n, weight_cols = weight.shape
    assert n % 8 == 0, "Weight rows must be divisible by 8"
    assert k == 2 * weight_cols, "Input K must match compressed weight columns"
    assert weight_cols % 8 == 0, "Compressed weight columns must be divisible by 8"
    assert input.stride(-1) == 1, "Expected input in row-major layout"
    expected_stride = input.stride(-2)
    for i in range(input.dim() - 3, -1, -1):
        expected_stride *= input.size(i + 1)
        assert input.stride(i) == expected_stride, "Expected input in row-major layout"
    assert weight.stride(1) == 1, "Expected weight in row-major layout"
    assert weight_meta.dtype == torch.uint8, "Expected uint8 weight metadata"
    assert weight_meta.dim() == 2, "Expected weight metadata to be 2D"
    assert weight_meta.stride(1) == 1, "Expected metadata in row-major layout"
    expected_meta_rows = max(((n + 63) // 64) * 64, 64)
    expected_meta_cols = max(((weight_cols // 4 + 15) // 16) * 16, 16)
    assert weight_meta.shape == (expected_meta_rows, expected_meta_cols), (
        "Unexpected weight metadata shape"
    )
    assert input_scale.shape == input.shape[:-1], "Unexpected input scale shape"
    assert input_scale.is_contiguous(), "Expected contiguous input scale"
    assert weight_scale.numel() == n, "Unexpected weight scale shape"
    assert weight_scale.is_contiguous(), "Expected contiguous weight scale"
    if bias is not None:
        assert bias.numel() == n, "Unexpected bias shape"
        assert bias.stride(0) == 1, "Expected contiguous bias"

    input_2d = input.reshape(-1, k)
    input_scale_1d = input_scale.flatten()
    weight_scale_1d = weight_scale.flatten()
    output = torch.empty(
        (input_2d.shape[0], n),
        dtype=output_dtype,
        device=input.device,
    )
    bias_arg = bias if bias is not None else output.reshape(-1)
    input_dtype_name = _float8_name(input.dtype)
    weight_dtype_name = _float8_name(weight.dtype)
    scale_dtype_name = _float_name(input_scale.dtype)
    output_dtype_name = _float_name(output_dtype)
    m = input_2d.shape[0]
    compiled = _compile_rowwise_scaled_linear_sparse_cutedsl(
        input_dtype_name,
        weight_dtype_name,
        scale_dtype_name,
        output_dtype_name,
        bias is not None,
        m % _WGMMA_INPUT_ROWS == 0,
        n % _CTA_WEIGHT_ROWS == 0,
        m % (2 * _WGMMA_INPUT_ROWS) == 0,
        weight_meta.shape[0] >= _CTA_WEIGHT_ROWS,
    )
    compiled(
        input=input_2d,
        input_scale=input_scale_1d,
        weight=weight,
        weight_meta=weight_meta,
        weight_scale=weight_scale_1d,
        bias=bias_arg,
        output=output,
        m_size=m,
        n_size=n,
        k_size=k,
        metadata_rows=weight_meta.shape[0],
        num_n_blocks=(n + _CTA_WEIGHT_ROWS - 1) // _CTA_WEIGHT_ROWS,
        raster_group_n=(
            _RASTER_GROUP_N
            if (n + _CTA_WEIGHT_ROWS - 1) // _CTA_WEIGHT_ROWS > 32
            else (n + _CTA_WEIGHT_ROWS - 1) // _CTA_WEIGHT_ROWS
        ),
        num_m_blocks=(m + _WGMMA_INPUT_ROWS - 1) // _WGMMA_INPUT_ROWS,
        stream=torch.cuda.current_stream(),
    )
    return output.reshape(*input.shape[:-1], n)
