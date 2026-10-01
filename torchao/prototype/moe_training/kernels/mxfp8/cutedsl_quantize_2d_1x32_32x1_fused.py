# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused 1x32 + 32x1 MXFP8 quantization -- one read of x, four outputs.

    x (total_M, N) bf16  +  offs (G,) int32, group sizes % 128 == 0
      ->  q_row (total_M, N)                     e4m3   [1x32 blocks along N]
          s_row (total_M, N//32)                 uint8, M-groups blocked
          q_col (N, total_M)                     e4m3   [32x1 blocks along M]
          s_col (N, total_M//32 + G*4)           uint8, K-groups blocked

LIMITATIONS
-----------
Hardware and runtime, enforced by the public wrapper in quant.py rather than
here: SM 10.x, CUDA 12.8+, and the CuTeDSL runtime (`nvidia-cutlass-dsl`,
`apache-tvm-ffi`). Without them this module raises NotImplementedError.

Input, all asserted in `quantize_fused`:

  * bf16 only.
  * x must be 2D, row-major contiguous, and 16-byte aligned.
  * total_M % 128 == 0 and N % 128 == 0. These are the tile dimensions, not a
    convenience -- one CTA owns a whole (128, 128) tile and neither edge is
    predicated.
  * offs must be 1D int32, contiguous, on device, holding ascending group END
    offsets along total_M.
  * block_size 32 and scaling_mode "rceil" only.

Every group size (consecutive difference in offs) must also be a multiple of
128; the s_col blocked layout addresses each group's slab with a stride
derived from that group's size, so a group that violates this would misplace
scales silently instead of failing. Checked with a device-side assert (thread
0 of each CTA, mirroring `cutedsl_quantize_2d_1x32.py` /
`cutedsl_quantize_2d_32x1.py`), not on the host, so it still surfaces as an
async CUDA error rather than a Python exception at the call site.
`offs[-1] < total_M` is allowed and is in fact the normal case; see
`quantize_fused` for what happens to that tail.

DESIGN
------
Parallelization. One CTA per (128, 128) tile of x: the grid is (N/128, M/128),
blockIdx.x selects the column slice and blockIdx.y the row slice. A CTA produces
everything its tile owns -- both quantized outputs and both sets of scales -- so
the two casts share one global read and CTAs never communicate. Block size is
picked in `_launch`: 512 threads when the whole grid is at most SMALL_GRID_CTAS
tiles, 256 otherwise, so a small grid still keeps enough threads in flight.
`threads` is a constexpr, and the two sizes are separately coded paths through
the kernel.

Staging. The tile is copied global->shared once with `cp.async`, 16 bytes per
thread per issue (4 issues at 512 threads, 8 at 256), then a single
`cp_async_wait_group(0)` and one barrier. SMEM holds the tile as u32 words, two
bf16 per word, in four 32-row groups. The group stride is 2056 words rather than
2048; that 8-word gap is what keeps the column pass free of bank conflicts.
Three views (`sW`, `sC256`, `sC512`) alias the same buffer -- the rowwise pass
reads it row-major, the colwise pass reads the same bytes strided down M, and
neither goes back to HBM.

Rowwise pass, 1x32 along N. Four consecutive lanes cooperate on one 32-element
block, 8 elements each. A lane folds its 8 with `max.u16x2` over sign-masked
bf16 -- non-negative floats order identically as unsigned integers, so that is
two elements per instruction -- and two butterfly shuffles finish the reduction
across the quad. `_scale` turns the amax into an E8M0 exponent plus a
reciprocal, `cvt.rn.satfinite.e4m3x2.bf16x2` scales and converts two elements at
a time, and the lead lane of each quad keeps the scale byte.

Colwise pass, 32x1 along M. The staged tile is re-read under a fresh
thread->data map, again at full CTA width, so neither pass strands the other. A
lane pair covers one 32-row block: each lane reduces 16 rows, and one butterfly
shuffle joins the halves. Since a u32 word holds two adjacent columns, the pair
produces two neighbouring 32x1 blocks at once -- hence two exponents per
iteration. `_scale` runs here with the colwise Inf convention, which differs
from the rowwise one.

Write-back. q_row and q_col go straight from registers to HBM. q_col is a
logical transpose and would scatter, so `prmt.b32` byte permutes de-interleave
the two columns first, leaving each store 16 contiguous bytes down M and fully
coalesced. Scales take a different route: each pass parks its bytes in a
512-byte SMEM buffer, and after a barrier 128 threads flush s_row while another
128 flush s_col, one coalesced store each. s_row lands at a flat offset --
128-aligned groups make its blocked layout the identity. s_col is the awkward
one: it scans `offs` for the group owning this tile and places its 512 bytes at
that group's dynamic stride, or zeroes those columns when no group owns the tile.
CTAs in the first row additionally clear the G*4-column over-allocation slab at
the tail of s_col while the rest of the grid is still computing.

COMPILATION
-----------
Compiled per concrete (total_M, N, G) -- all three are constant-folded into the
kernel and the result is cached in `_cache`, so a caller whose total_M varies
step to step pays a JIT compile on each new shape.
"""

from typing import Optional, Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    _cutedsl_runtime_available,
    _missing_cutedsl_runtime_packages,
)

_RUNTIME_OK = _cutedsl_runtime_available()


if _RUNTIME_OK:
    import cuda.bindings.driver as _cuda_drv
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import torch
    from cutlass import Int32, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
    from cutlass.cutlass_dsl import T, dsl_user_op

    from .cute_utils import validate_group_sizes

    TM = 128
    TN = 128
    SMALL_GRID_CTAS = 512

    @dsl_user_op
    def amax_acc(acc: Uint32, a: Uint32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    Uint32(acc).ir_value(loc=loc, ip=ip),
                    Uint32(a).ir_value(loc=loc, ip=ip),
                ],
                "{ .reg .b32 t;\n"
                "  and.b32 t, $2, 0x7fff7fff;\n"
                "  max.u16x2 $0, $1, t; }",
                "=r,r,r",
                has_side_effects=False,
                asm_dialect=0,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def cvt2(a: Uint32, b: Uint32, invx2: Uint32, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [
                    Uint32(a).ir_value(loc=loc, ip=ip),
                    Uint32(b).ir_value(loc=loc, ip=ip),
                    Uint32(invx2).ir_value(loc=loc, ip=ip),
                ],
                "{ .reg .b16 l, h; .reg .b32 t;\n"
                "  mul.rn.bf16x2 t, $1, $3;\n"
                "  cvt.rn.satfinite.e4m3x2.bf16x2 l, t;\n"
                "  mul.rn.bf16x2 t, $2, $3;\n"
                "  cvt.rn.satfinite.e4m3x2.bf16x2 h, t;\n"
                "  mov.b32 $0, {l, h}; }",
                "=r,r,r,r",
                has_side_effects=False,
                asm_dialect=0,
                loc=loc,
                ip=ip,
            )
        )

    def _prmt_op(sel):
        @dsl_user_op
        def f(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
            return Uint32(
                llvm.inline_asm(
                    T.i32(),
                    [
                        Uint32(a).ir_value(loc=loc, ip=ip),
                        Uint32(b).ir_value(loc=loc, ip=ip),
                    ],
                    "prmt.b32 $0, $1, $2, %d;" % sel,
                    "=r,r,r",
                    has_side_effects=False,
                    asm_dialect=0,
                    loc=loc,
                    ip=ip,
                )
            )

        return f

    prmt_even = _prmt_op(0x6420)
    prmt_odd = _prmt_op(0x7531)

    @cute.jit
    def _umax(a: Uint32, b: Uint32) -> Uint32:
        return a if a > b else b

    @cute.jit
    def _scale(bits: Uint32, degenerate_ib: Uint32, nosat_inf: cutlass.Constexpr[bool]):
        # Exact RCEIL for bf16 amax represented in the high half of an f32 word.
        w = bits + Uint32(0x1FFFFF)
        es = Int32(w >> Uint32(23)) - Int32(8)
        # cvt.rp.satfinite maps a subnormal descale to E8M0 byte 0.  Since
        # descale=amax/448, the exact BF16 cutoff is 1.75*2^-118 (0x04e0).
        e = es if bits >= Uint32(0x04E00000) else Int32(0)
        ib = Uint32((Int32(254) - e) << Int32(23))
        if e == Int32(0):
            ib = Uint32(0x3F800000) if bits == Uint32(0) else degenerate_ib
        # Inf/NaN. The amax above is an UNSIGNED max over sign-masked bf16 bit
        # patterns, which orders Inf and NaN as merely very large integers, so
        # without this a poisoned block would quantize to a healthy-looking 448.
        #
        # The two kernels being replaced DISAGREE here, so this does too, the
        # same way it carries their degenerate-block convention:
        #
        #   rowwise (cvt.rp.NOSAT.ue8m0x2.f32, cute_utils.py) maps Inf to 0xff,
        #     so Inf and NaN alike give scale 255 and a NaN reciprocal, and every
        #     element in the block comes out NaN.
        #   colwise (SATFINITE, mxfp8_quantize.cuh) CLAMPS Inf to 0xfe = 2^127
        #     and only NaN reaches 0xff. Its reciprocal is the E8M0 byte
        #     254 - scale, so scale 254 gives byte 0 = 2^-127 -- NOT the 0.0 that
        #     this function's (254 - e) << 23 would produce for e == 254.
        #
        # Neither branch is reachable from randn, so both need constructed input.
        if bits > Uint32(0x7F800000):
            e = Int32(255)
            ib = Uint32(0x7FC00000)
        elif bits == Uint32(0x7F800000):
            if cutlass.const_expr(nosat_inf):
                e = Int32(255)
                ib = Uint32(0x7FC00000)
            else:
                e = Int32(254)
                ib = Uint32(0x00800000)
        return e, ib

    @cute.kernel
    def _kernel(
        mX: cute.Tensor,
        mRQ: cute.Tensor,
        mCQ: cute.Tensor,
        mRS: cute.Tensor,
        mCS: cute.Tensor,
        mOffs: cute.Tensor,
        kblocks: cutlass.Constexpr,
        mblocks: cutlass.Constexpr,
        groups: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        kt, mt, _ = cute.arch.block_idx()

        # Validate group sizes are multiples of 128; a group that violates
        # this would otherwise silently misplace s_col scales (see module
        # docstring).
        if tid == 0:
            validate_group_sizes(mOffs)

        smem = utils.SmemAllocator()
        # The 8-u32 gap between 32-row groups removes column-pass bank conflicts.
        sbuf = smem.allocate_array(cutlass.Uint32, 4 * 2056, byte_alignment=16)
        srow = smem.allocate_array(cutlass.Uint32, 128, byte_alignment=16)
        scol = smem.allocate_array(cutlass.Uint32, 128, byte_alignment=16)
        sW = cute.make_tensor(
            sbuf, cute.make_layout((4, 2, 16, 16, 4), stride=(2056, 1028, 64, 4, 1))
        )
        sC256 = cute.make_tensor(
            sbuf, cute.make_layout((4, 2, 16, 32, 2), stride=(2056, 1028, 64, 2, 1))
        )
        sC512 = cute.make_tensor(
            sbuf, cute.make_layout((4, 2, 16, 64, 1), stride=(2056, 1028, 64, 1, 1))
        )
        sR32 = cute.make_tensor(srow, cute.make_layout(128))
        sC32 = cute.make_tensor(scol, cute.make_layout(128))
        sR8 = cute.make_tensor(
            cute.recast_ptr(srow, dtype=cutlass.Uint8), cute.make_layout(512)
        )
        sS8 = cute.make_tensor(
            cute.recast_ptr(scol, dtype=cutlass.Uint8), cute.make_layout(512)
        )

        if mt >= mblocks:
            # One extra CTA per group clears its 4-column over-allocation slab.
            if kt == 0:
                slab = mt - mblocks
                base = (
                    mblocks * TM // 32 * kblocks * TN + slab * 4 * kblocks * TN
                ) // 4
                i = tid
                while i < kblocks * TN:
                    mCS[base + i] = Uint32(0)
                    i += threads
        else:
            lane16 = tid % 16
            rbase = tid // 16
            kchunk = kt * 16 + lane16
            quad = lane16 // 4
            is_lead = (lane16 % 4) == 0
            sr_base = rbase * 16 + quad

            g2s = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
                cutlass.Uint32,
                num_bits_per_copy=128,
            )
            if cutlass.const_expr(threads == 256):
                for r in cutlass.range_constexpr(8):
                    lr = r * 16 + rbase
                    cute.copy(
                        g2s,
                        mX[(mt * TM + lr, kchunk, None)],
                        sW[(lr // 32, (lr % 32) // 16, lr % 16, lane16, None)],
                    )
            else:
                for r in cutlass.range_constexpr(4):
                    lr = r * 32 + rbase
                    cute.copy(
                        g2s,
                        mX[(mt * TM + lr, kchunk, None)],
                        sW[(lr // 32, (lr % 32) // 16, lr % 16, lane16, None)],
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            frg = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
            oq = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Uint32)
            if cutlass.const_expr(threads == 256):
                for r in cutlass.range_constexpr(8):
                    lr = r * 16 + rbase
                    cute.autovec_copy(
                        sW[(lr // 32, (lr % 32) // 16, lr % 16, lane16, None)], frg
                    )
                    acc = amax_acc(Uint32(0), frg[0])
                    acc = amax_acc(acc, frg[1])
                    acc = amax_acc(acc, frg[2])
                    acc = amax_acc(acc, frg[3])
                    v = _umax(acc << Uint32(16), acc & Uint32(0xFFFF0000))
                    v = _umax(v, Uint32(cute.arch.shuffle_sync_bfly(v, 1)))
                    v = _umax(v, Uint32(cute.arch.shuffle_sync_bfly(v, 2)))
                    e, ib = _scale(v, Uint32(0x7F000000), True)
                    invx2 = ib | (ib >> Uint32(16))
                    oq[0] = cvt2(frg[0], frg[1], invx2)
                    oq[1] = cvt2(frg[2], frg[3], invx2)
                    cute.autovec_copy(oq, mRQ[(mt * TM + lr, kchunk, None)])
                    if is_lead:
                        sR8[sr_base + ((r % 2) * 256 + (r // 2) * 4)] = cutlass.Uint8(e)
            else:
                for r in cutlass.range_constexpr(4):
                    lr = r * 32 + rbase
                    cute.autovec_copy(
                        sW[(lr // 32, (lr % 32) // 16, lr % 16, lane16, None)], frg
                    )
                    acc = amax_acc(Uint32(0), frg[0])
                    acc = amax_acc(acc, frg[1])
                    acc = amax_acc(acc, frg[2])
                    acc = amax_acc(acc, frg[3])
                    v = _umax(acc << Uint32(16), acc & Uint32(0xFFFF0000))
                    v = _umax(v, Uint32(cute.arch.shuffle_sync_bfly(v, 1)))
                    v = _umax(v, Uint32(cute.arch.shuffle_sync_bfly(v, 2)))
                    e, ib = _scale(v, Uint32(0x7F000000), True)
                    invx2 = ib | (ib >> Uint32(16))
                    oq[0] = cvt2(frg[0], frg[1], invx2)
                    oq[1] = cvt2(frg[2], frg[3], invx2)
                    cute.autovec_copy(oq, mRQ[(mt * TM + lr, kchunk, None)])
                    if is_lead:
                        sR8[sr_base + r * 4] = cutlass.Uint8(e)

            # Each lane pair owns rows of one 32x1 group; byte permutes produce
            # fully coalesced q_col stores despite the logical transpose.
            mg = (tid % 8) // 2
            half = tid % 2
            jq = tid // 8
            if cutlass.const_expr(threads == 256):
                cf = cute.make_rmem_tensor(
                    cute.make_layout((16, 2), stride=(2, 1)), Uint32
                )
                for i in cutlass.range_constexpr(16):
                    cute.autovec_copy(sC256[(mg, half, i, jq, None)], cf[(i, None)])

                a0 = amax_acc(Uint32(0), cf[0, 0])
                a1 = amax_acc(Uint32(0), cf[0, 1])
                b0 = amax_acc(Uint32(0), cf[1, 0])
                b1 = amax_acc(Uint32(0), cf[1, 1])
                for i in cutlass.range_constexpr(1, 8):
                    a0 = amax_acc(a0, cf[2 * i, 0])
                    a1 = amax_acc(a1, cf[2 * i, 1])
                    b0 = amax_acc(b0, cf[2 * i + 1, 0])
                    b1 = amax_acc(b1, cf[2 * i + 1, 1])
                a0 = amax_acc(a0, b0)
                a1 = amax_acc(a1, b1)
                a0 = amax_acc(a0, Uint32(cute.arch.shuffle_sync_bfly(a0, 1)))
                a1 = amax_acc(a1, Uint32(cute.arch.shuffle_sync_bfly(a1, 1)))

                oe = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                oo = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                mchunk = mt * 8 + mg * 2 + half
                ss_base = (jq % 8) * 64 + (jq // 8) * 4 + mg
                for w in cutlass.range_constexpr(2):
                    pk = a0 if w == 0 else a1
                    e0, ib0 = _scale(pk << Uint32(16), Uint32(0x7F000000), False)
                    e1, ib1 = _scale(pk & Uint32(0xFFFF0000), Uint32(0x7F000000), False)
                    invw = (ib0 >> Uint32(16)) | (ib1 & Uint32(0xFFFF0000))
                    for g in cutlass.range_constexpr(4):
                        ua = cvt2(cf[4 * g, w], cf[4 * g + 1, w], invw)
                        ub = cvt2(cf[4 * g + 2, w], cf[4 * g + 3, w], invw)
                        oe[g] = prmt_even(ua, ub)
                        oo[g] = prmt_odd(ua, ub)
                    cute.autovec_copy(oe, mCQ[(kt * TN + jq * 4 + 2 * w, mchunk, None)])
                    cute.autovec_copy(
                        oo, mCQ[(kt * TN + jq * 4 + 2 * w + 1, mchunk, None)]
                    )
                    if half == 0:
                        sS8[ss_base + (2 * w) * 16] = cutlass.Uint8(e0)
                        sS8[ss_base + (2 * w + 1) * 16] = cutlass.Uint8(e1)
            else:
                cf = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), Uint32)
                for i in cutlass.range_constexpr(16):
                    cf[i] = sC512[(mg, half, i, jq, 0)]

                acc = amax_acc(Uint32(0), cf[0])
                for i in cutlass.range_constexpr(1, 16):
                    acc = amax_acc(acc, cf[i])
                acc = amax_acc(acc, Uint32(cute.arch.shuffle_sync_bfly(acc, 1)))

                oe = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                oo = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Uint32)
                mchunk = mt * 8 + mg * 2 + half
                ss_base = (jq % 16) * 32 + (jq // 16) * 4 + mg
                e0, ib0 = _scale(acc << Uint32(16), Uint32(0x7F000000), False)
                e1, ib1 = _scale(acc & Uint32(0xFFFF0000), Uint32(0x7F000000), False)
                invw = (ib0 >> Uint32(16)) | (ib1 & Uint32(0xFFFF0000))
                for g in cutlass.range_constexpr(4):
                    ua = cvt2(cf[4 * g], cf[4 * g + 1], invw)
                    ub = cvt2(cf[4 * g + 2], cf[4 * g + 3], invw)
                    oe[g] = prmt_even(ua, ub)
                    oo[g] = prmt_odd(ua, ub)
                cute.autovec_copy(oe, mCQ[(kt * TN + jq * 2, mchunk, None)])
                cute.autovec_copy(oo, mCQ[(kt * TN + jq * 2 + 1, mchunk, None)])
                if half == 0:
                    sS8[ss_base] = cutlass.Uint8(e0)
                    sS8[ss_base + 16] = cutlass.Uint8(e1)

            cute.arch.barrier()
            if tid < 128:
                mRS[(mt * kblocks + kt) * 128 + tid] = sR32[tid]
            elif tid < 256:
                # Map this 512-byte col-scale tile into its dynamic group-major slab.
                m0 = mt * TM
                group_start = Int32(0)
                group_cols = Int32(0)
                found = Int32(0)
                prev = Int32(0)
                for g in cutlass.range_constexpr(groups):
                    end = Int32(mOffs[g])
                    if found == 0:
                        if m0 < end:
                            group_start = prev
                            group_cols = (end - prev) // 32
                            found = Int32(1)
                    prev = end
                if found == 0:
                    # UNCOVERED tile: m0 >= offs[-1], so no group owns it. A dropless
                    # MoE dispatcher sizes grad_out for worst-case padding and fills
                    # only a prefix, so this is the common case. torchao's K-groups
                    # swizzle never writes these scale columns and its buffer is
                    # new_zeros, so the contract here is ZERO.
                    #
                    # Falling through to the placement math below is doubly wrong:
                    # group_start=0 aims the store into GROUP 0's slab, and
                    # group_cols=0 kills the kt term so all kblocks CTAs race on one
                    # 512B line. That is what produced grad_norm=inf in TorchTitan.
                    #
                    # This tile owns exactly scale columns [m0//32, m0//32+4) == 4*N
                    # bytes; the kblocks n-tiles split them 512B each, so the fill is
                    # perfectly partitioned. Uncovered tiles together cover
                    # [offs[-1]//32, total_M//32) exactly, since every group size --
                    # hence offs[-1] -- is a multiple of 128.
                    zero_base = (m0 // 32) * kblocks * TN + kt * 512
                    mCS[zero_base // 4 + tid - 128] = Uint32(0)
                else:
                    local_cb = (m0 - group_start) // TM
                    byte_base = (
                        (group_start // 32) * kblocks * TN
                        + kt * (group_cols // 4) * 512
                        + local_cb * 512
                    )
                    mCS[byte_base // 4 + tid - 128] = sC32[tid - 128]

            # The first real CTA row clears the disjoint over-allocation tail while
            # the rest of the grid is still computing, avoiding G extra CTA rows.
            if mt == 0:
                tail_i = kt * threads + tid
                tail_words = groups * kblocks * TN
                tail_base = mblocks * kblocks * 128
                tail_stride = kblocks * threads
                while tail_i < tail_words:
                    mCS[tail_base + tail_i] = Uint32(0)
                    tail_i += tail_stride

    @cute.jit
    def _launch(tX, tOffs, tRQ, tRS, tCQ, tCS, stream):
        M = cutlass.const_expr(tX.shape[0])
        N = cutlass.const_expr(tX.shape[1])
        G = cutlass.const_expr(tOffs.shape[0])
        u32 = Uint32
        mX = cute.make_tensor(
            cute.recast_ptr(tX.iterator, dtype=u32),
            cute.make_layout((M, N // 8, 4), stride=(N // 2, 4, 1)),
        )
        mRQ = cute.make_tensor(
            cute.recast_ptr(tRQ.iterator, dtype=u32),
            cute.make_layout((M, N // 8, 2), stride=(N // 4, 2, 1)),
        )
        mCQ = cute.make_tensor(
            cute.recast_ptr(tCQ.iterator, dtype=u32),
            cute.make_layout((N, M // 16, 4), stride=(M // 4, 4, 1)),
        )
        mRS = cute.make_tensor(
            cute.recast_ptr(tRS.iterator, dtype=u32), cute.make_layout(M * N // 128)
        )
        mCS = cute.make_tensor(
            cute.recast_ptr(tCS.iterator, dtype=u32),
            cute.make_layout(N * (M // 32 + G * 4) // 4),
        )
        threads = cutlass.const_expr(
            512 if (M // TM) * (N // TN) <= SMALL_GRID_CTAS else 256
        )
        _kernel(mX, mRQ, mCQ, mRS, mCS, tOffs, N // TN, M // TM, G, threads).launch(
            grid=[N // TN, M // TM, 1], block=[threads, 1, 1], stream=stream
        )

    _cache = {}

    def _compile(x, offs, q_row, s_row, q_col, s_col):
        def w(t):
            return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)

        return cute.compile(
            _launch,
            w(x),
            w(offs),
            w(q_row),
            w(s_row),
            w(q_col),
            w(s_col),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    def run(x, offs, q_row, s_row, q_col, s_col):
        key = (x.shape, offs.shape)
        fn = _cache.get(key)
        if fn is None:
            fn = _compile(x, offs, q_row, s_row, q_col, s_col)
            _cache[key] = fn
        fn(
            x,
            offs,
            q_row,
            s_row,
            q_col,
            s_col,
            _cuda_drv.CUstream(int(torch.cuda.current_stream().cuda_stream)),
        )


def s_col_shape(total_M: int, N: int, num_groups: int) -> Tuple[int, int]:
    """Shape of the K-groups blocked colwise scale buffer.

    Over-allocated by G*4 columns: the true padded width depends on the group
    sizes, and computing it on the host would cost a D2H sync. The slack bytes
    are zeroed by this kernel.
    """
    return ceil_div(N, 128) * 128, total_M // 32 + num_groups * 4


def _aligned16(t: torch.Tensor) -> torch.Tensor:
    """`run` hands tensors to `from_dlpack(..., assumed_align=16)`.

    Torch's caching allocator returns 512-byte-aligned blocks, so this only ever
    bites on a view with an odd storage offset. Copying `offs` (a handful of
    ints) is free; doing the same to `x` would defeat the whole kernel, so `x`
    is asserted instead.
    """
    if t.data_ptr() % 16 == 0:
        return t
    return t.clone()


def quantize_fused(
    x: torch.Tensor,
    offs: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    q_row_out: Optional[torch.Tensor] = None,
    s_row_out: Optional[torch.Tensor] = None,
    q_col_out: Optional[torch.Tensor] = None,
    s_col_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both MXFP8 casts of grad_out from one pass over `x`.

    Args:
        x: (total_M, N) bfloat16, row-major. N % 128 == 0, total_M % 128 == 0.
        offs: (G,) int32 CUDA tensor of group END offsets along total_M,
              ascending, every group size a multiple of 128 -- checked with a
              device-side assert (see module docstring), which raises an async
              CUDA error rather than failing here. `offs[-1]` MAY be less than
              total_M -- a dropless MoE dispatcher sizes this buffer for
              worst-case padding and fills only a prefix, so an uncovered tail
              is the norm. Rows >= offs[-1] are still quantized into
              q_row/q_col/s_row (torchao's 1x32 kernel does the same), but
              their s_col scale columns are ZEROED to match torchao's
              K-groups swizzle.
        block_size: MX block size, 32 only.
        scaling_mode: "rceil" only (what TorchTitan runs).
        *_out: optional destination buffers (destination-passing, so a benchmark
            harness can hoist allocation out of the timed region). None of them
            need to be zeroed -- every byte of every output is written here.

    Returns:
        q_row (total_M, N) e4m3, s_row (total_M, N//32) uint8,
        q_col (N, total_M) e4m3, s_col (ceil(N,128)*128, total_M//32+G*4) uint8
    """
    if not _RUNTIME_OK:
        raise NotImplementedError(
            "missing CuTeDSL runtime package(s): "
            + ", ".join(_missing_cutedsl_runtime_packages())
        )

    assert x.is_cuda and offs.is_cuda, "x and offs must be CUDA tensors"
    assert x.dim() == 2 and x.is_contiguous(), "x must be 2D row-major contiguous"
    assert x.dtype == torch.bfloat16, "x must be bf16"
    assert block_size == 32, "only block_size=32 is supported"
    assert scaling_mode == "rceil", "only rceil is supported"
    assert offs.dtype == torch.int32 and offs.dim() == 1, "offs must be 1D int32"
    assert offs.is_contiguous(), "offs must be contiguous"

    total_M, N = x.shape
    assert N % 128 == 0, "N must be a multiple of 128"
    assert total_M % 128 == 0, "total_M must be a multiple of 128"
    assert x.data_ptr() % 16 == 0, "x must be 16-byte aligned"
    G = offs.numel()

    s_col_rows, s_col_cols = s_col_shape(total_M, N, G)

    if q_row_out is None:
        q_row_out = torch.empty(
            (total_M, N), device=x.device, dtype=torch.float8_e4m3fn
        )
    if s_row_out is None:
        s_row_out = torch.empty((total_M, N // 32), device=x.device, dtype=torch.uint8)
    if q_col_out is None:
        q_col_out = torch.empty(
            (N, total_M), device=x.device, dtype=torch.float8_e4m3fn
        )
    if s_col_out is None:
        s_col_out = torch.empty(
            (s_col_rows, s_col_cols), device=x.device, dtype=torch.uint8
        )

    run(x, _aligned16(offs), q_row_out, s_row_out, q_col_out, s_col_out)
    return q_row_out, s_row_out, q_col_out, s_col_out
