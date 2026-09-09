# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 grouped GEMM for AMD gfx950 (MI350X / MI355X), written in FlyDSL.

    out[group_g] = A[group_g] @ B[g]^T          A(M, K), B(E, N, K) -> (M, N)

Groups partition the token dim (the output rows) with device-resident, unequal
sizes; the contraction K is uniform across experts. This serves BOTH the MoE
forward and the dgrad -- the dgrad is the same grouping with grad_output as A
and the untransposed expert weight as B -- so one kernel covers both.

The group boundaries are read ON DEVICE, inside the kernel. Nothing here syncs
the offsets to the host, so real (imbalanced) MoE routing costs no D->H stall
and the compile key does not contain the token count.

Design, in one paragraph: a 256x256 output tile per workgroup with a 4-wave
2x2 grid, an 8-buffer LDS ping-pong feeding a fully unrolled K-walk, the f32
accumulators pinned in AGPRs across `v_mfma_scale_f32_16x16x128_f8f6f4`
(gfx950's scaled MFMA, which consumes the E8M0 block scales directly), a
one-shot pre-shuffle of both E8M0 scale planes into the layout the MFMA
fragments want, and `s_waitcnt vmcnt` counts derived from the issue order
rather than hand-transcribed. `NOTES` at the bottom of `grouped_mm` records the
shape constraints; unsupported shapes raise `NotImplementedError` rather than
silently producing wrong results.

Measured on one MI350X (gfx950), against `torch._grouped_mm` in bf16 on the
same shapes, GEMM only: geomean **1.93x**, min 1.59x, max 2.32x over the
36-shape grid in `benchmarks/prototype/moe_training/mxfp8/bench_flydsl_grouped_mm.py`.

Requires the `flydsl` package (https://github.com/ROCm/FlyDSL); like the FlyDSL
quantization kernels alongside it, the module imports cleanly without it and
the entry point raises a clear error.
"""

# ---------------------------------------------------------------------------
# Implementation notes
# ---------------------------------------------------------------------------
# The body is a port of a FlyDSL MXFP8 *wgrad* kernel (groups along the
# contracting dim). The port is small because both problems are `C = A @ B^T`
# with A and B row-major and the contraction running along the contiguous axis;
# only the meaning of the axes changes:
#
#                    wgrad                           this kernel
#   A                grad_out^T (N, M_TOTAL)         A      (M_tok, K)
#   B                input^T    (K, M_TOTAL)         B[g]   (N, K)
#   contraction      tokens, length m_g              K, uniform
#   groups partition the CONTRACTION                 the OUTPUT ROWS
#   per-group offset a column offset                 a row offset on A, a plane on B
#   output           (E, N, K) f32                   (M_tok, N) bf16
#
# So the group index moves out of the K-walk and into the tile's base
# addresses, and the ragged axis never reaches the pipeline: K is uniform
# across experts, so the K-walk cannot tell that the groups are ragged. What
# the raggedness does cost is (a) resolving block -> (group, row tile, col
# tile) on device from OFFS, and (b) an epilogue store masked on
# `active & (row < group_end)`, because a row tile may overhang its group and
# those rows hold this expert's weights applied to the next group's tokens.
#
# Two structural consequences of the short contraction:
#   * The K-loop is fully unrolled at compile time. The wgrad needed `scf.for`
#     because a 192-step constexpr loop hung the JIT; at 11-16 steps the rolled
#     form only costs scheduling freedom, and the loop-carried fragments and
#     scale chunk disappear.
#   * Pipeline fill and drain are a much larger fraction of the kernel (2 of 16
#     steps are prologue, 2 are tails), so prologue cost matters here in a way
#     it never did for the wgrad.
#
# The vmcnt accounting is DERIVED, not transcribed: with the loop unrolled the
# issue pattern is not periodic (late steps stop prefetching scales), so
# `_VmCounter` tracks issued vector-memory ops and each barrier waits for
# exactly "everything up to the op I depend on". Getting this wrong in the
# unsafe direction is silent -- too large a count under-waits and reads LDS the
# copy has not filled.

# NOTE: no `from __future__ import annotations` -- fx.struct needs real types.

import functools
import os
from typing import Optional

import torch

from torchao.utils import is_MI350

from . import flydsl_gemm_preshuffle as _sp
from .flydsl_launch import fast_launch
from .flydsl_utils import BLOCK_SIZE as SCALE_BLOCK
from .flydsl_utils import _flydsl_runtime_available, _missing_flydsl_runtime_packages

BLOCK_R = 256  # DEFAULT output rows (tokens) per tile; see pick_block_r
BLOCK_C_DEFAULT = 256  # output cols (N) per tile; 128 also supported, see pick_block_c
BLOCK_M = 128  # contraction elements per pipeline step (= BLOCK_K)


def pick_block_r(m_total: int, e: int) -> int:
    """Row tile height, from the AVERAGE group size. 256 unless groups are small.

    A row tile is charged for every row it covers, so a group of 64 tokens under
    a 256-row tile runs the MFMAs on 256 rows and discards three quarters of the
    result at the masked store. The waste is (BLOCK_R / m_g) and it is the single
    largest term at the small-M shapes: measured 4.6% of MXFP8 peak at
    m_g=64 (g32x64x2048x2048) against 20-25% where m_g >= 512.

    A narrower tile costs per-block efficiency, because the MFMA-per-ds_read
    ratio is NA*NB/(NA+NB) -- 2.0 at 4x4 (BLOCK_R=256), 1.33 at 2x4 (128), 0.8 at
    1x4 (64). So this is only worth taking when the overhang it removes is bigger
    than the ratio it gives up, i.e. when a group does not fill the wider tile.

    AVERAGE, not per-group, and that is forced: the group sizes live on the
    device and this kernel's whole design is not to sync on them (see
    `grouped_mm`). m_total and E are host-side shape metadata, so m_total // E
    costs nothing. Under real MoE routing the groups are within ~2x of the mean,
    so the average picks the right tile for most of them; a badly skewed batch
    gets a tile sized for its mean, which is still no worse than the fixed 256.

    THE CUTOFF IS MEASURED, NOT DERIVED, and it sits at 320 because that is
    where the evidence stops being one-sided. Pure overhang arithmetic says 128
    only pays below m_g=128, but it keeps winning up to a skewed ragged mean of
    269 (+44%), because the smaller tile also doubles the tile count and fills
    the machine. At m_avg=512 it is no longer a win in any consistent direction
    -- the same m_avg goes +5.2%, -2.6% and -14.0% depending only on K, N and E
    -- so 512 stays on the wide tile. Measured (median of 3, bitwise-identical
    output, TFLOP/s):

        m_avg  shape                     br=256   br=128   pick
           62  ragged [1,3,7,15,...]      254.8    366.2    128 (64 would give 430.3)
           64  g32x64x2048x2048           219.9    298.7    128 (64 would give 375.1)
          128  g8x128x4096x4096           461.6    595.7    128
          128  g16x128x4096x4096          408.0    536.9    128
          256  g16x256x2048x2048          628.5    614.0    128  (-2.3%, the one loss)
          256  g8x256x8192x8192           806.1    867.3    128
          269  ragged [100,333,777,...]   571.7    820.6    128
          512  g8x512x4096x4096           992.0   1044.1    256  (+5.2% forgone)
          512  g16x512x4096x4096         1036.6   1009.9    256
          512  g8x512x8192x2048           994.1    854.5    256  (128 costs 14%)
          696  ragged [813,1200,47,...]  1199.2   1073.7    256
         1024  g4x1024x2048x2048          771.1    682.4    256

    64 IS RETURNED AGAIN at m_avg <= 96, where it is the fastest tile (1.71x at
    g32x64, 1.69x at the [1,3,7,...] ragged). It used to compute wrong answers;
    the cause was `wait_barrier` waiting on vmcnt but never lgkmcnt, so an s2r
    fragment issued in one cluster and consumed in the next crossed the barrier
    outstanding while another wave overwrote its LDS buffer. Only the MFMAs in
    between covered the gap -- N_ACCUMS of them -- so the tiles with N_ACCUMS=4
    (this one, and BLOCK_C=128 with BLOCK_R=128) were the ones that broke. Fixed
    in `_fp8_gemm_utils.wait_barrier`; verified clean on all 10 shapes of
    scratchpad/dbg_br64.py on the development branch, including the K=8192
    N=8192 case that used to corrupt
    1510 elements.
    """
    if e <= 0:
        return 256
    m_avg = m_total // e
    if m_avg <= 96:
        return 64
    if m_avg <= 320:
        return 128
    return 256


def pick_tile(m_total: int, e: int, n: int) -> tuple:
    """(BLOCK_R, BLOCK_C) for this shape, shrinking both when the grid starves.

    `pick_block_r` and `pick_block_c` each answer a local question and neither
    looks at how many blocks the pair produces. That is fine everywhere except
    small N, where BLOCK_C=256 leaves ONE column tile and the row dim is the only
    parallelism left: K4096 x N256 tiles to 16 blocks on a 256-CU part, 6% of the
    machine, and it is the one PR shape still under 1x of the bf16 kernel.

    Halving a tile dimension doubles the block count, and at a starved grid that
    trade is strongly positive even though the narrower tile is worse per block
    (the MFMA-per-ds_read ratio argument in `pick_block_c`). Measured at
    K4096 x N256, M=4096 over 8 groups, bitwise-identical output:

        br    bc   tiles   TF/s
        256   256     16   145.4   (1.00x, the shipped pick)
        256   128     32   215.9   (1.48x)
        128   256     32   212.6   (1.46x)
        128   128     64   296.3   (2.04x)

    THE THRESHOLD IS ONE FULL WAVE, and the evidence for it had to be re-taken.
    The original block_c sweep -- "128 loses on 11 of 12 shapes", including 0.86x
    at g4x1024x2048x2048 -- was measured BEFORE the surplus-slot guard, when the
    launched grid was (ceildiv(M,BLOCK_R) + E) * n_c instead of the tile count,
    so both arms were partial-wave and the comparison never isolated occupancy.
    Post-guard the grid IS the tile count, the same shape goes +14%, and every
    partial-wave shape measured gains 11-14%. Do not resurrect the old numbers.
    """
    br = pick_block_r(m_total, e)
    bc = pick_block_c(n)
    if e <= 0 or n <= 0:
        return br, bc
    # Row tiles are per-group, so the average group is what the grid sees.
    tiles = max(1, e * ceildiv(max(m_total // e, 1), br)) * ceildiv(n, bc)
    if tiles >= _STARVED_TILES:
        return br, bc
    # Only a tile that is 256 in BOTH dims can be halved: halving one that is
    # already 128 lands on (128, 128), which is banned below. Measured on the
    # partial-wave shapes (TFLOP/s, all bitwise-correct, post-surplus-guard):
    #
    #   shape                 pick  (256,256)  (256,128)  (128,256)   taken
    #   g4x1024x2048x2048  256,256      823.4      938.1      929.9   +14%
    #   g4x512x4096x4096   256,256     1018.2     1164.8     1156.5   +14%
    #   g8x512x8192x2048   256,256     1110.8     1231.6     1267.7   +11%
    #   g8x128x4096x4096   128,256      522.8      618.8      730.1   -> (128,128)
    #   ragged [1,3,7,...] 128,256      269.4      335.5      386.2   -> narrower
    #
    # SHRINK ONE STEP AT A TIME, re-checking. (128,128) and BLOCK_R=64 were
    # banned until 2026-08-13 for computing wrong answers; the cause was
    # `wait_barrier` waiting on vmcnt and never lgkmcnt, so an s2r fragment
    # issued in one cluster and consumed in the next crossed the barrier still
    # outstanding while another wave overwrote its LDS buffer. The only thing
    # covering that gap was the MFMAs in between -- N_ACCUMS of them -- so
    # exactly the tiles with N_ACCUMS=4 broke. Fixed in `_fp8_gemm_utils`, and
    # both tiles verified clean on the repros that used to fail every time.
    # At K4096 x N256 the two-step shrink is worth 296.3 TF/s against 215.9 for
    # stopping at (256,128).
    tiles_at = lambda r, c: (
        max(1, e * ceildiv(max(m_total // e, 1), r)) * ceildiv(n, c)
    )
    if bc == 256 and n % 128 == 0:
        bc = 128
        if tiles_at(br, bc) >= _STARVED_TILES:
            return br, bc
    if br == 256:
        br = 128
        if tiles_at(br, bc) >= _STARVED_TILES:
            return br, bc
    # Third step, for the shapes that are STILL starved after both. K4096 x N256
    # is the case: it is bandwidth-bound with too few concurrent blocks to
    # saturate HBM, and throughput tracks achieved bandwidth, which tracks block
    # count. Measured there (bitwise-identical output):
    #
    #   tile      tiles  waves    TF/s   achieved TB/s
    #   (256,256)    16   0.06   147.4   0.63
    #   (256,128)    32   0.12   222.4   1.40
    #   (128,128)    64   0.25   302.8   2.51
    #   (64,128)    128   0.50   361.1   4.45
    #
    # TAKE THE LAST ROW AS +9%, NOT +19%. That 361.1 came from one back-to-back
    # sweep; an interleaved A/B repeated across three fresh processes puts
    # (64,128) at 1.073 / 1.095 / 1.110 of (128,128), i.e. ~336 TF/s against
    # ~305. One shape at +9% is ~+0.4% of the 24-shape geomean, which is inside
    # that harness's run-to-run band -- so this step is justified by the A/B and
    # will NOT be visible end to end. Do not go looking for it there.
    #
    # Split-K would be the textbook fix and is NOT it: the K-walk is only ~6us
    # of a 25us kernel there, so splitting K scaled the time LINEARLY with the
    # number of launches (S=2 -> 0.50x, S=4 -> 0.25x). More blocks per launch is
    # what this shape wants, not more launches.
    #
    # HALF the threshold, because this step reverses once bandwidth saturates.
    # A finer tile re-reads operands more times, so it only pays while there are
    # too few blocks to use the bus. K4096 x N512 shows the far side: it reaches
    # 4.58 TB/s at (128,128) already, and (64,128) buys 4.91 TB/s of MORE bytes
    # for LESS work -- 551.2 -> 398.6 TF/s. N256 is the near side, 2.51 -> 4.45
    # TB/s and 302.8 -> 361.1 TF/s. The two differ by whether 128 tiles is
    # already enough, so the gate is "still under half a wave after both steps".
    if br == 128 and tiles_at(br, bc) < _STARVED_TILES // 2:
        br = 64
    return br, bc


_STARVED_TILES = 256  # one full wave on MI350X's 256 CUs; see pick_tile

_guard_fallback_warned = set()

# (K, N, E, BLOCK_C, BLOCK_R) whose guarded kernel failed to TRACE. FlyDSL
# traces on first launch, not at compile, so this is discovered at the launch
# site and remembered here -- see `launch_for` / `launch_guarded`.
_guard_broken = set()


def _warn_guard_fallback(
    K, N, E, BLOCK_C, BLOCK_R, SWZ_XCD, SWZ_G, ZTAIL, PRESHUF, err
) -> None:
    """Say once, per shape, that the surplus-slot guard did not compile.

    Loud enough to notice (this shape is now paying for its surplus row-tile
    slots, up to 1.74x) but not per call, and never fatal.
    """
    key = (K, N, E, BLOCK_C, BLOCK_R, SWZ_XCD, SWZ_G, ZTAIL, PRESHUF)
    if key in _guard_fallback_warned:
        return
    _guard_fallback_warned.add(key)
    import logging

    logging.getLogger(__name__).warning(
        "[amd_titan] fwdgemm: surplus-slot guard failed to compile at "
        "K=%d N=%d E=%d BLOCK_C=%d BLOCK_R=%d (%s: %s); falling back to the "
        "unguarded kernel, which is correct but pays for its surplus slots.",
        K,
        N,
        E,
        BLOCK_C,
        BLOCK_R,
        type(err).__name__,
        str(err).splitlines()[0][:120],
    )


SUPERTILE_MAX_N_TILES = 16  # see pick_swizzle
SUPERTILE_ROWS = 4


def pick_swizzle(N: int, BLOCK_C: int) -> tuple:
    """(SWZ_XCD, SWZ_G) for the block -> tile rasterization. See `_rasterize`.

    A 4-row supertile when the output is at most 16 column tiles wide, plain
    row-major above that. The threshold is on ``n_c = ceildiv(N, BLOCK_C)``
    alone, and that is not a fit: the same sweep run in both shape regimes --
    E=32 / m_avg 4,096 (671B) and E=8 / m_avg 24,576 (16B) -- tracks to within a
    point at every n_c, so the effect belongs to the column-tile count and not to
    E or to tokens per expert. K=2048, 1x4 against off, arms interleaved, median
    of 3, every arm bitwise-identical:

        n_c   N       16B (E=8)   671B (E=32)
          6   1408      +6.7%        +7.6%
          8   2048      +5.5%        +4.7%
         11   2816      +4.6%        +4.4%
         16   4096      +2.2%        +2.3%
         28   7168      -0.5%        -1.1%

    WHY IT DECAYS WITH n_c. Row-major already hands the A stripe a run of `n_c`
    consecutive blocks. Once that run is longer than the blocks an XCD holds at
    once the locality is there without help, and grouping row tiles only breaks
    it up; when the run is short, the same handful of B stripes is cycled through
    and evicted once per row tile, which is what the supertile stops.

    1x8 was measured alongside and geomeans the same (+6.2% against 1x4's +6.2%
    over the n_c<=8 cells); 4 is taken because it rounds the slot count up less.
    At the 671B forward set this fires on `dgrad w2` (N=2048) only -- both
    DSV3-16B GEMMs (N=1408, N=2048) are inside it, so 16B gains more here.

    THE XCD GATHER IS NOT A DEFAULT, and the measurement is why. Undoing the
    hardware's round-robin so that stripe-sharing blocks land on one L2 is the
    obvious move and it LOSES: `8x1` against `off` on the 671B cells reads
    1616/1750, 1336/1411, 1410/1638 -- -4% to -8% on three of four, `dgrad w2`
    the only gain. Whatever the round-robin costs in locality it buys back in
    memory-channel spread. See `_rasterize` for the mechanism, and for why this
    was worth testing here when the wgrad's own reuse probe said 3.9%.

    ``TORCHAO_MXFP8_FLYDSL_GEMM_SWIZZLE`` pins an arm for an A/B: ``off`` for plain row-major, or
    ``<xcd>x<g>`` (e.g. ``8x8``). Unset means this policy.
    """
    v = os.environ.get("TORCHAO_MXFP8_FLYDSL_GEMM_SWIZZLE", "auto").lower()
    if v not in ("auto", ""):
        if v in ("off", "0", "none"):
            return 0, 1
        xcd, _, g = v.partition("x")
        return int(xcd), int(g or 1)
    if ceildiv(N, BLOCK_C) <= SUPERTILE_MAX_N_TILES:
        return 0, SUPERTILE_ROWS
    return 0, 1


def pick_block_c(N: int) -> int:
    """Column tile width. Always 256 -- the narrow tile was measured and lost.

    The motivation for a narrow tile is real: a tile that overhangs N is not
    free, because the MFMAs run on the overhang and the results are discarded at
    the store. At K=2048, M=196608, N=1408 (6x256 = 1536 columns) and N=1536 take
    the SAME 0.83 ms, so the 9.1% overhang is paid in full, and the kernel's
    29.8% of peak on useful FLOPs is really 32.3% of the hardware.

    1408 = 2^7 x 11 and a legal BLOCK_C is 64 x N_TILES_B, so 128 is the only
    sane width that divides it. Measured (median of 3, interleaved):

        K=2048 N=1408   256: 0.831 ms 1365 TF/s   128: 0.979 ms 1158   0.849x
        K=1408 N=2048   256: 0.833 ms 1361 TF/s   128: 1.034 ms 1097   0.806x
        K=2048 N=2048   256: 1.105 ms 1493 TF/s   128: 1.326 ms 1244   0.833x

    So the narrow tile removes the whole overhang at N=1408 and is still 15%
    slower. Why: per K-step a wave issues 4*NA*NB MFMAs against 4*(NA+NB) LDS
    reads, so the MFMA-per-ds_read ratio is NA*NB/(NA+NB) -- 2.0 at 4x4, 1.33 at
    4x2. Hardware efficiency drops 32.3% -> 25.1%, a 22% loss to save 9% of work.

    There is no better tile available: the accumulators already fill all 256
    AGPRs at NA*NB = 16 per quadrant, so no shape with a higher ratio fits. The
    NA=8, NB=2 corner (ratio 1.6, and exactly 160 KB of LDS) interpolates to
    ~28% hardware, still below the 29.8% useful the wide tile delivers.

    Conclusion: the overhang at N=1408 is not recoverable by retiling. Pass
    ``block_c=128`` explicitly to re-run the comparison.
    """
    return BLOCK_C_DEFAULT


if _flydsl_runtime_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import arith, const_expr, range_constexpr, rocdl
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec

    from . import flydsl_buffer_ops as buffer_ops

    def _uniform(v):
        """Assert wave-uniformity to the backend: `v` is the same in every lane.

        `rocdl.readfirstlane` IS that assertion -- one `v_readfirstlane_b32`,
        and the result is SGPR-resident, so downstream uses are SALU instead of
        a per-use re-extraction. Only call it on a genuinely uniform value; on a
        divergent one it silently keeps lane 0. See the wgrad kernel's copy for why this
        matters (m0 setup per LDS-DMA fill, and buffer-soffset waterfalls).
        """
        return fx.Int32(rocdl.readfirstlane(res=T.i32, src=arith._to_raw(v)))

    from .flydsl_gemm_utils import (
        G2SLoader,
        S2RLoader,
        compute_global_swizzle,
        make_fp8_buffer_tensor,
        pack_i32x4_i32x8,
        swizzle_128,
        wait_barrier,
    )

    # Interleave each quadrant's 16 MFMAs with the g2s/s2r loads of the NEXT
    # fragment so they co-issue in the MFMA execute shadow. Worth +6.6% in wgrad
    # once its scale loads stopped saturating the L1 address path. Set
    # TORCHAO_MXFP8_FLYDSL_GEMM_INTERLEAVE=0 for the plain cluster.
    _INTERLEAVE = os.environ.get("TORCHAO_MXFP8_FLYDSL_GEMM_INTERLEAVE", "1") != "0"

    # Occupancy hint. 128 KB of LDS already caps this at one workgroup per CU,
    # so the attribute only bounds the per-wave register budget -- it cannot
    # legitimately change results. It USED to (wrong output on every shape at
    # 2, correct at 1), and the cause was NOT a missing wait: forcing every
    # pipeline barrier to vmcnt(0) left the corruption unchanged, so the
    # `_VmCounter` accounting was exonerated by experiment. The real
    # mechanism, read from the compiled ISA:
    #
    #   At waves_per_eu=2 the backend grants the kernel only the 4 AGPRs the
    #   `=a` inline-asm constraint forces (192 arch VGPRs + 4 AGPRs, against
    #   184 + 64 at 1), so instead of pinning each accumulator in its own AGPR
    #   quad the register allocator homes them in VGPRs and shuttles them
    #   through a[0:3]: `v_accvgpr_write` of SrcC immediately before every
    #   MFMA and `v_accvgpr_read` of the result 1-2 instructions after it.
    #   Both copies sit inside the MFMA's software-managed hazard windows, and
    #   the hazard recognizer cannot see that the producer is an MFMA -- it is
    #   an opaque asm blob -- so the s_nops it would insert for a real MFMA
    #   are missing. Proof both ways: padding the asm with s_nops on both
    #   sides makes waves_per_eu=2 bitwise-clean, and so does the shipped fix
    #   below, which keeps the accumulators out of the shuttle entirely.
    #
    # THE FIX is `amdgpu-agpr-alloc` (see AGPR_NEED in `_compile`): reserving
    # the accumulator count in AGPRs restores the waves_per_eu=1 allocation
    # shape -- accumulators resident, `v_accvgpr_read` only in the epilogue,
    # where instruction distance covers the hazard exactly as it always has at
    # 1. The attribute is attached only when _WAVES_PER_EU != 1, so the
    # shipping setting compiles byte-identically to what was always shipped.
    _WAVES_PER_EU = int(os.environ.get("TORCHAO_MXFP8_FLYDSL_GEMM_WAVES_PER_EU", "1"))

    # ENERGY, not cycles: this kernel runs pinned to the 1000 W package cap, so
    # `time = energy / 1000 W` and instructions removed from the hot body buy
    # clock (EXPERIMENTS.md S4). Two hot-loop values are wave-uniform by
    # construction and the backend cannot prove it: the LDS destination of every
    # `buffer_load ... offen lds` (it lives in m0, an SGPR -- so a VGPR-resident
    # base costs `v_readfirstlane_b32` + `s_mov_b32 m0` before EVERY fill; this
    # kernel's ISA carried 527 of them against 512 fills), and the pre-shuffled
    # scale load's soffset (a VGPR-resident SGPR operand makes the backend wrap
    # the load in a waterfall loop -- 32 of those). Both are asserted uniform
    # with `_uniform`.
    _SCALAR_ADDR = os.environ.get("TORCHAO_MXFP8_FLYDSL_GEMM_SCALAR_ADDR", "1") != "0"

    def _mfma_scale_agpr(a, b, sa, sb, acc, opsel=0):
        """fp8 16x16x128 scaled MFMA with the accumulator pinned in AGPR (=a,...,0)
        so the f32x4 accumulates in place and the compiler does not shuffle it
        between AGPR slots. a/b are i32x8 (K=128 fp8); sa/sb are broadcast-i32
        E8M0 scales (the e8m0 in byte 0). cbsz:0 blgp:0."""
        # The scale operands are i32 and the MFMA reads ONE byte of each,
        # chosen by op_sel/op_sel_hi. Naming neither is bit-identical to
        # `op_sel:[0,0,0] op_sel_hi:[0,0,0]` on gfx950 (checked against
        # llvm-mc: same 16-byte encoding), i.e. byte 0 -- so the callers do
        # NOT need to broadcast the e8m0 across all four bytes, and the
        # `* 0x01010101` that used to do so was 16 dead v_mul per K-step.
        # To select another byte the modifiers must precede cbsz/blgp; the
        # reverse order is rejected by the assembler.
        # `opsel` (0..3) is which byte of each i32 scale operand the MFMA takes.
        # The modifiers must PRECEDE cbsz/blgp -- the reverse order is rejected --
        # and opsel 0 deliberately emits the bare form, which llvm-mc encodes
        # byte-for-byte the same as `op_sel:[0,0,0] op_sel_hi:[0,0,0]`, so the
        # default path stays exactly the instruction this kernel always shipped.
        _osel = (
            ""
            if opsel == 0
            else f"op_sel:[{opsel & 1},{opsel & 1},0] "
            f"op_sel_hi:[{(opsel >> 1) & 1},{(opsel >> 1) & 1},0] "
        )
        asm = (
            "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 "
            f"{_osel}cbsz:0 blgp:0"
        )
        return _llvm.inline_asm(
            Vec.make_type(4, fx.Float32),
            [
                arith._to_raw(a),
                arith._to_raw(b),
                arith._to_raw(sa),
                arith._to_raw(sb),
                arith._to_raw(acc),
            ],
            asm,
            "=a,v,v,v,v,0",
            has_side_effects=True,
        )

    class _VmCounter:
        """Issue-order bookkeeping for `s_waitcnt vmcnt(n)`.

        vmcnt retires IN ORDER, so "wait until op X has landed" is spelled
        "allow at most (ops issued after X) to be outstanding". Every vector
        memory op the kernel issues is counted here; `mark()` records a
        watermark right after the ops a later barrier will depend on, and
        `wait(mark)` emits the barrier with the exact count.

        Getting this wrong in the unsafe direction is silent: too LARGE a count
        under-waits and reads LDS the copy has not filled yet. Deriving it from
        the issue order removes the chance to mis-transcribe a constant.
        """

        def __init__(self):
            self.n = 0

        def issue(self, k):
            self.n += k
            return self.n

        def mark(self):
            return self.n

        def wait(self, mark):
            wait_barrier(self.n - mark)

    # ── Block -> logical tile rasterization ─────────────────────────────────
    # Row-major (SWZ_XCD=0, SWZ_G=1) is the historical order and the default.
    #
    # WHY A SWIZZLE CAN MATTER HERE AND DID NOT FOR THE WGRAD. Traffic per FLOP
    # is fixed by the tile (256 FLOP/byte at 256x256), so the only free variable
    # is L2 hit rate. The wgrad's operand stripes are (BR or BC) x M_TOTAL --
    # 33 MB each at the 671B shapes -- so no scheduling makes them co-resident,
    # and `probe_reuse.py` measured the whole available effect at 3.9%. The
    # FORWARD's stripes are (BR or BC) x K: 1.84 MB at K=7168, 0.5 MB at K=2048.
    # Blocks that share one fit in a 4 MB XCD-local L2 together, and per K-step
    # the shared window is only BLOCK_R x 128 = 32 KB. So the reuse the wgrad did
    # not have is physically available here, and it was worth asking for.
    #
    # THE ANSWER WAS HALF YES. The supertile pays at small n_c and the XCD gather
    # loses outright; `pick_swizzle` has both tables and the default that came out
    # of them. Read that before reaching for either. What follows is the mechanism.
    #
    # Two independent transforms, both constexpr so an unset knob traces to the
    # identity:
    #
    #   SWZ_XCD=8   Undo the hardware's round-robin. Workgroup b runs on XCD
    #               (b + phase) % 8 (measured, `probe_xcd.py`), so consecutive
    #               block ids -- the ones that share an A stripe under row-major
    #               -- land on eight DIFFERENT L2s. Gathering each residue class
    #               into a contiguous run of logical ids puts them back on one.
    #               Invariant to `phase`, which is nonzero and set by the
    #               preceding dispatch: a rotation only relabels which XCD gets
    #               which run.
    #   SWZ_G=g     Rasterize column-major inside a supertile of `g` row tiles,
    #               so the g blocks issued together share one B stripe (one
    #               expert's 256 weight columns) instead of touching g different
    #               ones.
    #
    # The map must be a BIJECTION on [0, n_blocks) or tiles go uncomputed. The
    # XCD gather is one on the head `8 * (n_blocks // 8)` and the identity on the
    # tail; the supertile needs `n_slots` to be a multiple of `g`, which the host
    # guarantees by rounding it up (the extra slots are surplus and exit at the
    # `active` guard, so they cost a dispatch and nothing else).
    def _rasterize(bid, n_blocks, n_c_tiles, SWZ_XCD, SWZ_G):
        lid = bid
        if const_expr(SWZ_XCD > 1):
            q = n_blocks // fx.Int32(SWZ_XCD)
            head = q * fx.Int32(SWZ_XCD)
            gathered = (lid % fx.Int32(SWZ_XCD)) * q + lid // fx.Int32(SWZ_XCD)
            lid = arith.select(lid < head, gathered, lid)
        if const_expr(SWZ_G > 1):
            per_super = fx.Int32(SWZ_G) * n_c_tiles
            sup = lid // per_super
            rem = lid % per_super
            lid = (
                sup * fx.Int32(SWZ_G) + rem % fx.Int32(SWZ_G)
            ) * n_c_tiles + rem // fx.Int32(SWZ_G)
        return lid

    def _compile(
        K: int,
        N: int,
        E: int,
        BLOCK_C: int,
        BLOCK_R: int = BLOCK_R,
        preshuf: bool = False,
        GUARD: bool = True,
        SWZ_XCD: int = 0,
        SWZ_G: int = 1,
        ZTAIL: bool = False,
    ):
        """Compile a RAGGED forward grouped GEMM for one (K, N, E) shape.

        Deliberately absent from the key: the token count and every group size.
        Both are runtime. Under real MoE routing they change every step, and a
        shape-keyed compile leaks a JIT compile plus a retained GPU module per
        step per layer -- the same defect the dim1 cast had, where it cost 618x.
        The even-groups parent keyed on (K, N, M_G, M_TOTAL, BLOCK_C) and so
        recompiled per token count; only balanced routing hid it.
        """
        assert K % BLOCK_M == 0, f"K ({K}) must be a multiple of {BLOCK_M}"
        assert BLOCK_C in (128, 256), f"BLOCK_C {BLOCK_C} not supported"
        # 64/128/256 give N_TILES_A of 1/2/4. Below 64 a half is under one MFMA
        # tile (16 rows x 2 wave-rows) and the row structure stops holding.
        assert BLOCK_R in (64, 128, 256), f"BLOCK_R {BLOCK_R} not supported"

        K_ITERS = K // BLOCK_M
        assert K_ITERS >= 4, (
            f"K {K} gives {K_ITERS} steps; need >= 4 (2 prologue + 2 tails)"
        )

        SCALE_I32_ROW = K // 128  # i32 of e8m0 per operand row

        N_TILES_A = BLOCK_R // 4 // 16
        N_TILES_B = BLOCK_C // 4 // 16
        N_ACCUMS = N_TILES_A * N_TILES_B
        N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

        # AGPRs the accumulators need to stay RESIDENT: 4 quadrants x N_ACCUMS
        # f32x4 tiles. Matches what the allocator picks on its own at
        # waves_per_eu=1 (64 at the 128x128 tile, 32 at 64x128, 256 at
        # 256x256). Passed as `amdgpu-agpr-alloc` when _WAVES_PER_EU != 1 --
        # see the block comment at _WAVES_PER_EU for why that is load-bearing
        # for correctness, not an optimization. Where the reservation cannot
        # fit the tighter budget (the 256x256 tile at waves_per_eu=2), the
        # backend prints "failed to meet occupancy target" and falls back to
        # occupancy 1, which is exactly the correct compile.
        AGPR_NEED = 16 * N_ACCUMS

        # A dwordx2 scale load needs an even element index. The index is
        # lane*SCALE_I32_ROW + (row_base*SCALE_I32_ROW) + k with k always even
        # (a chunk starts on an even K-step) and no per-group K offset here, so
        # the row stride alone decides. K=2048 -> 16 (paired); K=1408 -> 11 (two
        # dword loads instead; same cache lines, one extra instruction).
        SC_PAIR = SCALE_I32_ROW % 2 == 0

        # ── Pre-shuffled scales (see _scale_preshuffle.py) ──
        # The cooperative load hands lane L the dword of ROW L, which is not the
        # row its MFMA fragment needs -- hence the ds_bpermute and the byte
        # extract, 48 VALU per K-step against 16 MFMAs. Pre-shuffled, one dwordx4
        # per operand-half puts all four of a lane's tiles in its own registers,
        # PACK K-steps ride the four bytes of each, and `op_sel` picks between
        # them for free. Because this K-walk is fully unrolled the K index is a
        # Python int, so PACK=4 costs nothing in structure and HALVES the number
        # of scale loads.
        #
        # A half is 64 rows over 4 tiles by construction of the slab layout, so
        # the pre-shuffled body only exists at the (256,256) tile.
        PRESHUF = preshuf and N_TILES_A == 4 and N_TILES_B == 4
        SC_PACK = _sp.PACK if PRESHUF else 1
        # scale vm ops per chunk: 4 dwordx4 (one per half) pre-shuffled, else the
        # cooperative pair/dword loads over 2 K-steps.
        N_SC_OPS = 4 if (PRESHUF or SC_PAIR) else 8
        N_CHUNKS = (K_ITERS + SC_PACK - 1) // SC_PACK if PRESHUF else (K_ITERS + 1) // 2
        SC_STEP = SC_PACK if PRESHUF else 2  # K-steps a chunk covers
        _SC_ISSUE_PHASE = 2 if PRESHUF else 0  # keeps a 2-step prefetch horizon
        _SP_K128P = _sp.ceildiv(K // 128, _sp.PACK)
        _SP_B_SPE = _sp.b_slabs_per_expert(N, BLOCK_C)
        _SP_B_SLABS = E * _SP_B_SPE

        LDS_BLOCK_R = BLOCK_R // 2  # each wave-dim half owns BLOCK_R/2 rows
        LDS_BLOCK_C = BLOCK_C // 2

        def _interleave_plan(n_mfma, n_g2s, n_tiles):
            """Where to slot each load into the cluster's MFMA stream.

            Returns three lists indexed by MFMA position: the g2s steps, the
            first-half s2r tiles and the second-half s2r tiles to issue just
            before that MFMA. Generated rather than written out because the
            counts move with the tile -- BLOCK_C=256 is 16 MFMAs against 4 g2s
            steps and 4x2 s2r reads, BLOCK_C=128 is 8 against 4 and 2x2.

            Computed HERE, outside the kernel body, on purpose: FlyDSL's AST
            rewriter turns an `if` inside a nested kernel-body function into a
            separate `__then_N` function that cannot see the enclosing closure,
            so schedule logic has to be plain Python that runs before tracing.
            The kernel body then walks the plan with no branches at all.
            """
            acts = []
            for r in range(max(n_g2s, n_tiles)):
                if r < n_g2s:
                    acts.append(("g", r))
                if r < n_tiles:
                    acts.append(("s0", r))
                    acts.append(("s1", r))
            g_at = [[] for _ in range(n_mfma)]
            s0_at = [[] for _ in range(n_mfma)]
            s1_at = [[] for _ in range(n_mfma)]
            for n_, act in enumerate(acts):
                # Spread the loads evenly across the MFMA stream. The clamp is
                # load-bearing at NA=1 (BLOCK_R=64), where 9 acts share only 4
                # MFMA slots: the last act lands at round(9*4/10) = round(3.6) =
                # 4, one past the end. `round` can exceed the floor whenever the
                # ratio's fraction is >= .5, so the clamp -- not the ratio -- is
                # what guarantees every load keeps a slot. It is a no-op at the
                # 4x4/2x4/4x2 tiles, where the ratio never reaches n_mfma - 0.5.
                pos = min(n_mfma - 1, int(round((n_ + 1) * n_mfma / (len(acts) + 1))))
                {"g": g_at, "s0": s0_at, "s1": s1_at}[act[0]][pos].append(act[1])
            return g_at, s0_at, s1_at

        MFMA_SEQ = [(i, j) for i in range(N_TILES_A) for j in range(N_TILES_B)]
        # Keyed by (g2s steps, s2r tiles): clusters 1 and 4 push A and pull B,
        # clusters 2 and 3 the other way round. Keys collide harmlessly when the
        # tile is square.
        PLANS = {
            (N_TILES_A, N_TILES_B): _interleave_plan(
                len(MFMA_SEQ), N_TILES_A, N_TILES_B
            ),
            (N_TILES_B, N_TILES_A): _interleave_plan(
                len(MFMA_SEQ), N_TILES_B, N_TILES_A
            ),
        }
        a_lds_size = LDS_BLOCK_R * BLOCK_M
        b_lds_size = LDS_BLOCK_C * BLOCK_M

        @fx.struct
        class SharedStorage:
            A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
            B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
            B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

        @flyc.kernel(known_block_size=[256, 1, 1])
        def kernel_fwd(
            A: fx.Tensor,
            B: fx.Tensor,
            OUT: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            OFFS: fx.Tensor,
            a_slabs: fx.Int32,
            n_c_tiles: fx.Int32,
            out_m: fx.Int32,
            out_n: fx.Int32,
            n_blocks: fx.Int32,
        ):
            F8_IR_t = fx.Float8E4M3FN.ir_type

            # ── Block -> (group, row tile, col tile), resolved ON DEVICE ──
            # Groups partition the OUTPUT ROWS here, not the contraction, so
            # unlike the wgrad kernel nothing about the K-walk changes: K is
            # uniform across experts. Only this mapping and the epilogue mask
            # care that the groups are ragged.
            bid = fx.block_idx.x
            lid = _rasterize(ArithValue(bid), n_blocks, n_c_tiles, SWZ_XCD, SWZ_G)
            slot = lid // n_c_tiles  # which row tile overall
            c_base = (lid % n_c_tiles) * fx.Int32(BLOCK_C)

            offs_rsrc = buffer_ops.create_buffer_resource(
                OFFS, max_size=False, num_records_bytes=E * 4
            )
            # Uniform (SGPR) loads: every lane in the block wants the same
            # boundaries, so this is E scalar loads, not E per-lane loads.
            ends = [
                ArithValue(
                    buffer_ops.buffer_load(
                        offs_rsrc, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
                    )
                )
                for i in range(E)
            ]
            starts = [ArithValue(fx.Int32(0))] + ends[:-1]

            # Walk the groups accumulating row tiles until the slot lands in one.
            # E is constexpr, so this unrolls to E selects on the SALU against a
            # K-walk of many iterations -- the same O(E) prologue the wgrad
            # kernel pays. `active` is false for slots past the last group's
            # tiles, which predicates those blocks off in the epilogue.
            g = ArithValue(fx.Int32(0))
            r_base = ArithValue(fx.Int32(0))
            m_start = ArithValue(fx.Int32(0))
            m_end = ArithValue(fx.Int32(0))
            active = fx.Int32(0) > fx.Int32(0)  # false
            cum = ArithValue(fx.Int32(0))
            # A's pre-shuffled slabs are PER GROUP -- a group boundary is only
            # padded to AMD_MXFP8_PAD_MULTIPLE, which can be 32, and a 32-aligned
            # start would straddle two slabs of a plane-aligned grid. `pre` is
            # this group's slab base; the pre-pass lays them out with the same
            # scan. E more SALU adds in a prologue that already runs E selects.
            pre = ArithValue(fx.Int32(0))
            cum_sl = ArithValue(fx.Int32(0))
            # range_constexpr, not range: the AST rewriter turns a plain `range`
            # into a device-side scf.for, whose induction variable is an
            # ArithValue and cannot index the `ends`/`starts` Python lists. This
            # loop must unroll at trace time.
            for i in range_constexpr(E):
                m_i = ends[i] - starts[i]
                t_i = (m_i + fx.Int32(BLOCK_R - 1)) // fx.Int32(BLOCK_R)
                hit = (slot >= cum) & (slot < cum + t_i)
                g = arith.select(hit, fx.Int32(i), g)
                r_base = arith.select(hit, (slot - cum) * fx.Int32(BLOCK_R), r_base)
                m_start = arith.select(hit, starts[i], m_start)
                m_end = arith.select(hit, ends[i], m_end)
                active = active | hit
                pre = arith.select(hit, cum_sl, pre)
                cum = cum + t_i
                cum_sl = cum_sl + t_i * fx.Int32(BLOCK_R // 64)

            # `active` is exactly `slot < cum`, so its complement needs no
            # negation op -- which is what ZTAIL below wants.
            z_slot = slot - cum
            zrow0 = ends[E - 1] + z_slot * fx.Int32(BLOCK_R)

            # A is offset by the group's first token row, B by the expert's
            # weight plane. Both are plain row offsets at the same stride K --
            # the "dynamic per-group stride" problem the MXFP8-MoE post
            # describes does not arise on this axis.
            row0 = m_start + r_base
            wrow0 = ArithValue(g) * fx.Int32(N) + c_base

            # ── SURPLUS SLOTS EXIT HERE, BEFORE ANY GLOBAL TRAFFIC ──
            # The host cannot know how many row tiles the groups actually need
            # -- that is sum_g ceildiv(m_g, BLOCK_R), and the m_g live on the
            # device -- so it launches the upper bound, ceildiv(M, BLOCK_R) + E.
            # The slack is real: at g8x512x4096x4096 it is 24 slots launched for
            # 16 needed, and since n_blocks = n_slots * n_c that is 384 blocks
            # where 256 would do. On 256 CUs that is the difference between one
            # wave and two, which is why the surplus cost 1.74x there and
            # 1.12-1.68x across the shapes -- far more than its 33% share of the
            # blocks, because it is wave quantization and not just wasted bytes.
            #
            # `active` was already computed for the store mask; consulting it
            # here instead skips the K-walk entirely. It is BLOCK-UNIFORM (it
            # depends only on block_idx and on scalar loads of OFFS), so every
            # thread takes the same branch and the barriers inside stay legal.
            # `GUARD` is a plain Python bool from the compile key, so the tracer
            # resolves this at TRACE time: True leaves `proceed` dynamic and the
            # rewriter builds the scf.if; False makes it a Python True and the
            # body is traced unconditionally, reproducing the pre-guard kernel
            # byte for byte. One copy of the body, two kernels. The fallback
            # exists because the guard does not compile everywhere -- see
            # `cached_launch`.
            proceed = active if GUARD else True
            if proceed:
                lds = fx.SharedAllocator().allocate(SharedStorage).peek()
                a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
                a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
                b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
                b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1

                lane_id = fx.thread_idx.x % 64
                # A ternary, not an `if`: inside a traced function an `if`
                # cannot BIND a name the code after it reads.
                wave_id = (
                    _uniform(fx.thread_idx.x // 64)
                    if _SCALAR_ADDR
                    else fx.thread_idx.x // 64
                )
                wave_i = wave_id // 2
                wave_j = wave_id % 2

                m_lane = lane_id % fx.Int32(16)
                k_grp = lane_id // fx.Int32(16)
                kshift = k_grp * fx.Int32(8)
                mask_ff = fx.Int32(0xFF)

                A0_gl = row0 * fx.Int32(K)
                A1_gl = (row0 + fx.Int32(LDS_BLOCK_R)) * fx.Int32(K)
                B0_gl = wrow0 * fx.Int32(K)
                B1_gl = (wrow0 + fx.Int32(LDS_BLOCK_C)) * fx.Int32(K)

                gA = make_fp8_buffer_tensor(A, F8_IR_t)
                gB = make_fp8_buffer_tensor(B, F8_IR_t)
                a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
                b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

                # BOUND THE SCALE DESCRIPTORS. `max_size=True` sets num_records to
                # 0xFFFFFFFF, so an out-of-range e8m0 read is NOT clamped by the
                # hardware -- it touches unmapped memory and faults. This kernel
                # over-provisions row tiles (`n_slots = ceildiv(M, BLOCK_R) + E`) and
                # relies on `active` to predicate the surplus blocks off in the
                # epilogue, so a scale read issued before that predicate goes past
                # the plane whenever the group boundaries leave a slot mapping past
                # the last group.
                #
                # Measured, not theorised: with AMD_MXFP8_PAD_MULTIPLE=256 (which
                # puts every group boundary on a 256 row-tile boundary) DSV3-16B at
                # lbs4/seq8192 dies with hipErrorIllegalAddress every run, and a
                # backend bisect pins it here -- AMD_MXFP8_GEMM_BACKEND=triton passes
                # while quant=triton and wgrad=triton both still fault. At the
                # default pad multiple of 32 the same fault appears every few runs.
                #
                # The wgrad body already carries this fix (_kernel.py); this
                # kernel was ported from it before the fix landed and never got it.
                # A bounded descriptor returns 0 for an out-of-range read, and 0 as
                # an e8m0 is 2**-127, which underflows the product exactly as the
                # epilogue mask intends.
                #
                # A is (M, K/32) with M a runtime value; B is (E, N, K/32), all
                # constexpr, so its bound folds to a constant.
                # Pre-shuffled, a plane is (slabs x packed-K x 64 lanes x 4
                # tiles) of i32; raw, it is rows x K/32 bytes. Both are bounded
                # for the reason above.
                _sa_n = (
                    a_slabs * fx.Int32(_SP_K128P * 64 * 4 * 4)
                    if PRESHUF
                    else out_m * fx.Int32(K // SCALE_BLOCK)
                )
                sa_bytes = arith.index_cast(T.i64, arith.index_cast(T.index, _sa_n))
                sa_rsrc = buffer_ops.create_buffer_resource(
                    A_scale, max_size=False, num_records_bytes=sa_bytes
                )
                sb_rsrc = buffer_ops.create_buffer_resource(
                    B_scale,
                    max_size=False,
                    num_records_bytes=(
                        _SP_B_SLABS * _SP_K128P * 64 * 4 * 4
                        if PRESHUF
                        else E * N * (K // SCALE_BLOCK)
                    ),
                )

                gl_off_a = compute_global_swizzle(
                    lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
                )
                gl_off_b = compute_global_swizzle(
                    lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
                )

                a_g2s = G2SLoader(a_div, gl_off_a, N_TILES_A, F8_IR_t, wave_id)
                b_g2s = G2SLoader(b_div, gl_off_b, N_TILES_B, F8_IR_t, wave_id)
                a_s2r = S2RLoader(wave_i, N_TILES_A)
                b_s2r = S2RLoader(wave_j, N_TILES_B)

                vm = _VmCounter()

                # ── MXFP8 scales: cooperative wide load + ds_bpermute (wgrad v3.3) ──
                # Each operand-half needs the e8m0 byte of 64 consecutive rows (4 MFMA
                # tiles x 16 rows) at this K-step; one i32 per row covers 128 K, of
                # which lane group k_grp takes byte k_grp and broadcasts it to 4 bytes.
                #
                # The naive form -- one buffer_load per (half, tile) -- issues 16 dword
                # loads per K-step that touch the SAME 16 cache lines four times over,
                # because lanes m_lane..m_lane+48 share an address. That is the same L1
                # request count as all of the step's data traffic, and it, not the
                # matrix core, sets the pace (measured in wgrad: stubbing the scales
                # out took it 822 -> 1486 TFLOP/s).
                #
                # Instead: ONE cooperative load per half, lane L taking row
                # (half_base + L), so 64 lanes cover 64 rows with no duplicate address,
                # widened to a dwordx2 spanning two K-steps. 8x fewer scale L1
                # requests. Each lane then pulls the tile it actually needs with
                # ds_bpermute (lane L from lane t*16 + L%16) -- LDS crossbar, not HBM.
                #
                # A half spans N_TILES*16 rows, so 64 lanes cover it exactly at
                # N_TILES=4 and cover it twice over at N_TILES=2 (BLOCK_C=128). The
                # duplicate half of that load is wasted address bandwidth but stays
                # ONE instruction, and the rows it touches are the next 32 of the same
                # weight plane -- already-warm cache lines, or an in-bounds read the
                # bpermute never selects.
                assert N_TILES_A <= 4 and N_TILES_B <= 4, "a half must fit in 64 lanes"

                sa0_base = row0 + wave_i * fx.Int32(N_TILES_A * 16)
                sa1_base = (
                    row0 + fx.Int32(LDS_BLOCK_R) + wave_i * fx.Int32(N_TILES_A * 16)
                )
                sb0_base = wrow0 + wave_j * fx.Int32(N_TILES_B * 16)
                sb1_base = (
                    wrow0 + fx.Int32(LDS_BLOCK_C) + wave_j * fx.Int32(N_TILES_B * 16)
                )

                # Only the row term is per-lane; the half base and the K index are
                # wave-uniform and ride the buffer instruction's SGPR soffset, so the
                # address costs no per-step VALU and exactly one VGPR.
                sc_voff = lane_id * fx.Int32(SCALE_I32_ROW)
                sc_sbase = [
                    (_rs, _base * fx.Int32(SCALE_I32_ROW * 4))
                    for _rs, _base in (
                        (sa_rsrc, sa0_base),
                        (sa_rsrc, sa1_base),
                        (sb_rsrc, sb0_base),
                        (sb_rsrc, sb1_base),
                    )
                ]
                # Pre-shuffled: the address is (slab, packed-K, lane, tile) and
                # the only per-lane term is `lane*4`. A's slab counts from its
                # group's base `pre` and B's from its expert's, so both are exact
                # whatever the group start is. r_base and c_base are multiples of
                # BLOCK_R/BLOCK_C, so every half base is a multiple of 64.
                sp_voff = lane_id * fx.Int32(4)
                _a_sl = pre + r_base // fx.Int32(64)
                _b_sl = ArithValue(g) * fx.Int32(_SP_B_SPE) + c_base // fx.Int32(64)
                sp_slab = [
                    (sa_rsrc, _a_sl + wave_i * fx.Int32(N_TILES_A * 16 // 64)),
                    (
                        sa_rsrc,
                        _a_sl
                        + fx.Int32(LDS_BLOCK_R // 64)
                        + wave_i * fx.Int32(N_TILES_A * 16 // 64),
                    ),
                    (sb_rsrc, _b_sl + wave_j * fx.Int32(N_TILES_B * 16 // 64)),
                    (
                        sb_rsrc,
                        _b_sl
                        + fx.Int32(LDS_BLOCK_C // 64)
                        + wave_j * fx.Int32(N_TILES_B * 16 // 64),
                    ),
                ]
                sc_perm = [
                    fx.Int32(t_ * 64) + m_lane * fx.Int32(4)
                    for t_ in range_constexpr(max(N_TILES_A, N_TILES_B))
                ]
                # Tiles per half, in the order the chunk stores them: A0, A1, B0, B1.
                sc_tiles = (N_TILES_A, N_TILES_A, N_TILES_B, N_TILES_B)

                def issue_chunk(k_i32):
                    """Cooperative scale loads for K-steps (k_i32, k_i32+1).

                    Returns 2 raw i32 per operand-half, flat: [a0_lo, a0_hi, a1_lo, ...].
                    A chunk for a K-step past the end reads the next row's scales (or
                    out of bounds, which the buffer resource returns as 0) and is never
                    consumed -- cheaper than predicating the last prefetch.
                    """
                    # SINGLE EXIT, and no conditional BINDING. FlyDSL rewrites
                    # this function's AST, which costs an `if` two things: an
                    # early `return` inside one runs the branch but does not leave
                    # the function, and a name bound inside one is not visible
                    # after it. Every conditional here mutates a list that already
                    # exists, or is a ternary.
                    out = []
                    if PRESHUF:
                        # One dwordx4 per operand-half: this lane's four tiles,
                        # with K-steps k_i32 .. k_i32+PACK-1 in the four bytes.
                        _kp = k_i32 // SC_PACK  # k_i32 is a Python int here
                        for _rs, _sl in sp_slab:
                            _soff = (
                                _sl * fx.Int32(_SP_K128P) + fx.Int32(_kp)
                            ) * fx.Int32(64 * 4 * 4)
                            _soff = _uniform(_soff) if _SCALAR_ADDR else _soff
                            v = buffer_ops.buffer_load(
                                _rs,
                                sp_voff,
                                vec_width=4,
                                dtype=T.i32,
                                soffset_bytes=_soff,
                            )
                            for t_ in range_constexpr(4):
                                out.append(
                                    buffer_ops.vec_extract(
                                        v, static_position=[t_], dynamic_position=[]
                                    )
                                )
                    for _rs, _sb in () if PRESHUF else sc_sbase:
                        soff = _sb + fx.Int32(k_i32 * 4)
                        if SC_PAIR:
                            v = buffer_ops.buffer_load(
                                _rs,
                                sc_voff,
                                vec_width=2,
                                dtype=T.i32,
                                soffset_bytes=soff,
                            )
                            out.append(
                                buffer_ops.vec_extract(
                                    v, static_position=[0], dynamic_position=[]
                                )
                            )
                            out.append(
                                buffer_ops.vec_extract(
                                    v, static_position=[1], dynamic_position=[]
                                )
                            )
                        else:
                            out.append(
                                buffer_ops.buffer_load(
                                    _rs,
                                    sc_voff,
                                    vec_width=1,
                                    dtype=T.i32,
                                    soffset_bytes=soff,
                                )
                            )
                            out.append(
                                buffer_ops.buffer_load(
                                    _rs,
                                    sc_voff,
                                    vec_width=1,
                                    dtype=T.i32,
                                    soffset_bytes=soff + fx.Int32(4),
                                )
                            )
                    vm.issue(N_SC_OPS)
                    return out

                def use_chunk(chunk, p):
                    """Redistribute + consume the chunk's K-step `p` -> (sa0, sa1, sb0, sb1)."""
                    groups = []
                    if PRESHUF:
                        # Nothing to redistribute, nothing to unpack and -- unlike
                        # the wgrad -- nothing to mask: this kernel's ragged axis
                        # is the output rows, not the contraction, so no scale is
                        # ever out of window. A slice.
                        for h_ in range_constexpr(4):
                            grp = []
                            for t_ in range_constexpr(4):
                                grp.append(chunk[4 * h_ + t_])
                            groups.append(grp)
                    for h_ in () if PRESHUF else range_constexpr(4):
                        src = chunk[2 * h_ + p]
                        grp = []
                        for t_ in range_constexpr(sc_tiles[h_]):
                            v = rocdl.ds_bpermute(res=T.i32, index=sc_perm[t_], src=src)
                            # No `* 0x01010101`: op_sel is pinned to byte 0, so
                            # the broadcast was 16 dead v_mul per K-step.
                            grp.append((ArithValue(v) >> kshift) & mask_ff)
                        groups.append(grp)
                    return groups

                def mma(a, b, c, sa, sb, osel=0):
                    for i in range_constexpr(N_TILES_A):
                        for j in range_constexpr(N_TILES_B):
                            idx = i * N_TILES_B + j
                            c[idx] = _mfma_scale_agpr(
                                a[i], b[j], sa[i], sb[j], c[idx], opsel=osel
                            )
                    return c

                zero = Vec.filled(4, 0.0, fx.Float32)
                c00 = [zero] * N_ACCUMS
                c01 = [zero] * N_ACCUMS
                c10 = [zero] * N_ACCUMS
                c11 = [zero] * N_ACCUMS

                def _lds_swizzle(s2r):
                    out = []
                    for row_off in range_constexpr(s2r.n_tiles):
                        row = (
                            s2r.wave_idx * fx.Int32(s2r.n_tiles * 16)
                            + fx.Int32(row_off * 16)
                            + (lane_id % fx.Int32(16))
                        )
                        swz = []
                        for ii in range_constexpr(2):
                            col = (lane_id // fx.Int32(16)) * fx.Int32(16) + fx.Int32(
                                ii * 64
                            )
                            r_, c_ = swizzle_128(row, col)
                            swz.append(r_ * fx.Int32(BLOCK_M) + c_)
                        out.append(swz)
                    return out

                def _cluster_plain(
                    lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb, osel=0
                ):
                    g2s.load(lds_dst, k_off)
                    vm.issue(g2s.n_load_steps)
                    rt = s2r.load(lds_src)
                    c = mma(a, b, c, sa, sb, osel)
                    return c, rt

                def _cluster_il(
                    lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb, osel=0
                ):
                    # Interleave this quadrant's MFMAs with the g2s and s2r loads of
                    # the NEXT fragment, so the loads co-issue in the MFMA execute
                    # shadow. Mirrors fp8_gemm_4wave's _interleaved_cluster; the MFMA
                    # is scaled + AGPR-pinned here. The schedule comes from
                    # `_interleave_plan` (computed before tracing), so this walks it
                    # with no branches -- see that function on why.
                    swz = _lds_swizzle(s2r)
                    rt = [None] * s2r.n_tiles
                    half = [None] * s2r.n_tiles
                    g_at, s0_at, s1_at = PLANS[(g2s.n_load_steps, s2r.n_tiles)]

                    for idx in range_constexpr(len(MFMA_SEQ)):
                        for gi in range_constexpr(len(g_at[idx])):
                            g2s.load_one(lds_dst, k_off, g_at[idx][gi])
                        for si in range_constexpr(len(s0_at[idx])):
                            t_ = s0_at[idx][si]
                            half[t_] = s2r.load_one(lds_src, swz[t_][0])
                        for si in range_constexpr(len(s1_at[idx])):
                            t_ = s1_at[idx][si]
                            rt[t_] = pack_i32x4_i32x8(
                                half[t_], s2r.load_one(lds_src, swz[t_][1])
                            )
                        i_, j_ = MFMA_SEQ[idx]
                        c[i_ * N_TILES_B + j_] = _mfma_scale_agpr(
                            a[i_],
                            b[j_],
                            sa[i_],
                            sb[j_],
                            c[i_ * N_TILES_B + j_],
                            opsel=osel,
                        )

                    vm.issue(g2s.n_load_steps)
                    return c, rt

                _cluster = _cluster_il if _INTERLEAVE else _cluster_plain

                # ── Prologue: pre-fill the 8-buffer LDS pipeline (2 K-steps) ──
                # Chunk 0 goes first so it is the oldest thing in flight and the
                # prologue barriers retire it for free.
                chunks = [None] * N_CHUNKS
                chunks[0] = issue_chunk(0)

                a_g2s.load(a_cur0, A0_gl + 0 * BLOCK_M)
                mk_ac0 = vm.issue(N_TILES_A)
                b_g2s.load(b_cur0, B0_gl + 0 * BLOCK_M)
                mk_bc0 = vm.issue(N_TILES_B)
                b_g2s.load(b_cur1, B1_gl + 0 * BLOCK_M)
                vm.issue(N_TILES_B)
                a_g2s.load(a_cur1, A1_gl + 0 * BLOCK_M)
                mk_a = vm.issue(N_TILES_A)

                a_g2s.load(a_next0, A0_gl + 1 * BLOCK_M)
                vm.issue(N_TILES_A)
                b_g2s.load(b_next0, B0_gl + 1 * BLOCK_M)
                mk_b = vm.issue(N_TILES_B)
                b_g2s.load(b_next1, B1_gl + 1 * BLOCK_M)
                vm.issue(N_TILES_B)
                a_g2s.load(a_next1, A1_gl + 1 * BLOCK_M)
                mk_c = vm.issue(N_TILES_A)

                vm.wait(mk_ac0)
                a0 = a_s2r.load(a_cur0)
                vm.wait(mk_bc0)
                b0 = b_s2r.load(b_cur0)

                # ── Main K-steps 0 .. K_ITERS-3, fully unrolled ──
                # Each step consumes the LDS buffers holding step k, loads step k+2
                # into them, and reads step k+1's leading fragments. Which watermark
                # each barrier waits on follows from the ping-pong depth:
                #   * clusters 1-2 read the halves written by step k-2's SECOND half,
                #   * clusters 3-4 read the halves written by step k-1's FIRST half.
                # The two prologue groups stand in for steps -2 and -1: `mk_a` closes
                # the k=0 group (step 0's second-half operands), `mk_b` sits right
                # after b_next0 (step 1's leading fragments) and `mk_c` closes the
                # k=1 group (step 1's second-half operands).
                bufs = (
                    a_cur0,
                    a_cur1,
                    a_next0,
                    a_next1,
                    b_cur0,
                    b_cur1,
                    b_next0,
                    b_next1,
                )
                sec = {-2: mk_a, -1: mk_c}  # second-half watermark, by step index
                fst = {-1: mk_b}  # first-half watermark, by step index

                for kk in range_constexpr(K_ITERS - 2):
                    ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                    k2 = (kk + 2) * BLOCK_M

                    # The scales for this step came off HBM a full step-pair ago; the
                    # only HBM touch is the prefetch for the pair two K-steps out --
                    # the same horizon the data ping-pong runs at.
                    vm.wait(sec[kk - 2])
                    # A chunk covers SC_STEP K-steps and is issued exactly two
                    # steps before its first use -- the same horizon the data
                    # ping-pong runs at, and the reason `vm.wait(sec[kk])` two
                    # steps later already retires it. Pre-shuffled that is 4 steps
                    # per chunk, so this fires at kk % 4 == 2 instead of every
                    # other step: HALF the scale loads for the same dwords.
                    ci = kk // SC_STEP
                    if kk % SC_STEP == _SC_ISSUE_PHASE and ci + 1 < N_CHUNKS:
                        chunks[ci + 1] = issue_chunk((ci + 1) * SC_STEP)
                    osel = kk % SC_STEP if PRESHUF else 0
                    sa0, sa1, sb0, sb1 = use_chunk(chunks[ci], kk % SC_STEP)

                    c00, b1 = _cluster(
                        ac0, a_g2s, A0_gl + k2, b_s2r, bc1, a0, b0, c00, sa0, sb0, osel
                    )
                    c01, a1 = _cluster(
                        bc0, b_g2s, B0_gl + k2, a_s2r, ac1, a0, b1, c01, sa0, sb1, osel
                    )
                    fst[kk] = vm.mark()

                    vm.wait(fst[kk - 1])
                    c10, a0 = _cluster(
                        bc1, b_g2s, B1_gl + k2, a_s2r, an0, a1, b0, c10, sa1, sb0, osel
                    )
                    c11, b0 = _cluster(
                        ac1, a_g2s, A1_gl + k2, b_s2r, bn0, a1, b1, c11, sa1, sb1, osel
                    )
                    sec[kk] = vm.mark()

                    bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

                a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = (
                    bufs
                )

                # ── Tail step K_ITERS-2: drain, no more g2s ──
                k_tail0 = K_ITERS - 2
                vm.wait(sec[k_tail0 - 2])
                _os = k_tail0 % SC_STEP if PRESHUF else 0
                sa0, sa1, sb0, sb1 = use_chunk(
                    chunks[k_tail0 // SC_STEP], k_tail0 % SC_STEP
                )
                b1 = b_s2r.load(b_cur1)
                c00 = mma(a0, b0, c00, sa0, sb0, _os)
                a1 = a_s2r.load(a_cur1)
                c01 = mma(a0, b1, c01, sa0, sb1, _os)
                vm.wait(fst[k_tail0 - 1])
                a0 = a_s2r.load(a_next0)
                c10 = mma(a1, b0, c10, sa1, sb0, _os)
                b0 = b_s2r.load(b_next0)
                c11 = mma(a1, b1, c11, sa1, sb1, _os)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

                # ── Tail step K_ITERS-1 ──
                k_tail1 = K_ITERS - 1
                vm.wait(sec[k_tail1 - 2])
                _os = k_tail1 % SC_STEP if PRESHUF else 0
                sa0, sa1, sb0, sb1 = use_chunk(
                    chunks[k_tail1 // SC_STEP], k_tail1 % SC_STEP
                )
                b1 = b_s2r.load(b_cur1)
                a1 = a_s2r.load(a_cur1)
                c00 = mma(a0, b0, c00, sa0, sb0, _os)
                c01 = mma(a0, b1, c01, sa0, sb1, _os)
                c10 = mma(a1, b0, c10, sa1, sb0, _os)
                c11 = mma(a1, b1, c11, sa1, sb1, _os)

                # ── Epilogue: bf16 into out(M, N); the MFMA already applied the scales ──
                # Storing f32 and converting outside would cost a full extra pass over
                # M*N (553 MB at the e2e forward shape) -- more than half the kernel.
                out_m_i = arith.index_cast(T.index, out_m)
                out_n_i = arith.index_cast(T.index, out_n)
                nbytes = arith.index_cast(T.i64, out_m_i * out_n_i * fx.Index(2))
                o_rsrc = buffer_ops.create_buffer_resource(
                    OUT, max_size=False, num_records_bytes=nbytes
                )
                base_row = row0 + wave_i * fx.Int32(N_TILES_A * 16)
                base_col = c_base + wave_j * fx.Int32(N_TILES_B * 16)
                oob = out_m * out_n

                # Row offsets in ELEMENTS for the four rows a lane owns inside a
                # tile: r_*out_n == row*out_n + e*out_n, and `out_n` is a runtime
                # value, so hoisting the three multiples out of every loop turns
                # 4 multiply-adds per tile into 1 multiply and 3 adds.
                # Explicit loops, not comprehensions: this function's AST is
                # rewritten by FlyDSL and the plain-Python forms are what the
                # rest of this kernel uses.
                _e_off = []
                for _e in range_constexpr(4):
                    _e_off.append(out_n * fx.Int32(_e))

                def store_group(frag, br, bc):
                    for ti in range_constexpr(N_TILES_A):
                        row = br + fx.Int32(ti * 16) + k_grp * fx.Int32(4)
                        # The two ROW bounds do not depend on `tj` -- the column
                        # tile does not move the row -- so they are computed once
                        # per (ti, e) here instead of once per (ti, tj, e) inside
                        # the loop below: 32 compares per epilogue instead of 128.
                        # `active` folds in here too, being invariant in all three.
                        row_ok = []
                        for e_ in range_constexpr(4):
                            _r = row + fx.Int32(e_)
                            row_ok.append(active & (_r < m_end) & (_r < out_m))
                        row_base = row * out_n
                        pend = []
                        for tj in range_constexpr(N_TILES_B):
                            col = bc + fx.Int32(tj * 16) + m_lane
                            col_ok = col < out_n
                            vec = Vec(frag[ti * N_TILES_B + tj])
                            tile_base = row_base + col
                            for e in range_constexpr(4):
                                # `row + e < m_end` is the ragged part. A row tile may
                                # overhang its group, in which case the overhanging
                                # rows hold this expert's weights applied to the
                                # NEXT group's tokens -- correct arithmetic, wrong
                                # expert -- so they must not be stored. `active`
                                # kills whole slots past the last group's tiles.
                                # Reading those A rows is harmless: each output
                                # element is one A row dotted with one B column, so
                                # nothing leaks across rows, and any fp8 NaN in the
                                # pad region past m_total lands only in rows we drop.
                                ok = row_ok[e] & col_ok
                                off = arith.select(ok, tile_base + _e_off[e], oob)
                                pend.append((e, tj, off, vec[e].to(fx.BFloat16)))

                        # ISSUE ORDER. A wave64 store here covers 4 rows x 32 B: lanes 0-15
                        # are 16 consecutive columns of one row, and `k_grp`
                        # (lane // 16) moves the row, not the column. So the
                        # bytes a single store touches are four 32-byte pieces
                        # of four DIFFERENT rows -- four partial writes into
                        # four 128-byte lines.
                        #
                        # `tj` moves the column by 16 elements = 32 B, so the
                        # four `tj` of one (ti, e) are the four ADJACENT 32-byte
                        # pieces of the same rows: issued back to back they
                        # complete whole 128-byte lines and the L2 can combine
                        # them. `e` moves the row instead, so the shipped
                        # ti -> tj -> e order issued four different rows in a
                        # row and left every line partial. Emitting e-major
                        # costs nothing -- same instructions, same operands,
                        # only the order of side effects changes.
                        pend.sort(key=lambda t: (t[1], t[0]))
                        for _e, _tj, _off, _val in pend:
                            buffer_ops.buffer_store(_val, o_rsrc, _off)

                store_group(c00, base_row, base_col)
                store_group(c01, base_row, base_col + fx.Int32(LDS_BLOCK_C))
                store_group(c10, base_row + fx.Int32(LDS_BLOCK_R), base_col)
                store_group(
                    c11,
                    base_row + fx.Int32(LDS_BLOCK_R),
                    base_col + fx.Int32(LDS_BLOCK_C),
                )

            # ── ZTAIL: the SURPLUS slots zero the output tail ──────────────
            # Rows at and past `OFFS[-1]` belong to no group, so no tile writes
            # them, and `grouped_mm` used to cover that by allocating the output
            # with `torch.zeros` -- a memset of the WHOLE M x N plane. At the
            # 671B forward shapes that plane is 1.88 GB and the memset is 10.7%
            # of the `w2` GEMM (measured: 2.466 ms with `out=` against 2.76 ms
            # allocating). But the untouched region is only M - OFFS[-1] rows,
            # which the padded dispatcher caps at E * PAD_MULTIPLE -- 8,192 of
            # 139,264 here. So zero THAT instead, from the blocks that were
            # already being launched and thrown away.
            #
            # The kernel launches `ceildiv(M, BLOCK_R) + E` row-tile slots
            # because the true count, `sum_g ceildiv(m_g, BLOCK_R)`, is a
            # device-side quantity. The slack has always been dead weight. Here
            # slot `cum + z` takes tail rows `[OFFS[-1] + z*BLOCK_R, +BLOCK_R)`,
            # and there are provably enough of them: t_i <= m_i/BLOCK_R + 1 gives
            # cum < OFFS[-1]/BLOCK_R + E, so
            #   surplus = ceildiv(M, BLOCK_R) + E - cum > (M - OFFS[-1])/BLOCK_R,
            # and being an integer it is therefore >= ceildiv(tail, BLOCK_R).
            # Slots past that write nothing: `r_ < out_m` masks them, which is
            # also what handles the ragged case where the tail is empty.
            if const_expr(ZTAIL):
                if slot >= cum:
                    znb = arith.index_cast(
                        T.i64,
                        arith.index_cast(T.index, out_m)
                        * arith.index_cast(T.index, out_n)
                        * fx.Index(2),
                    )
                    zo_rsrc = buffer_ops.create_buffer_resource(
                        OUT, max_size=False, num_records_bytes=znb
                    )
                    zoob = out_m * out_n
                    tid = fx.thread_idx.x
                    # 256 threads cover one BLOCK_C-wide row (or two, at
                    # BLOCK_C=128), so a store instruction writes BLOCK_C
                    # contiguous bf16 -- 512 bytes, whole cache lines. Scalar
                    # per lane, exactly like the main epilogue: a vector<Nxbf16>
                    # operand does not survive the backend here ("do not know
                    # how to scalarize this operator's operand").
                    zero_b = Vec.filled(4, 0.0, fx.Float32)[0].to(fx.BFloat16)
                    z_rows = 256 // BLOCK_C
                    zcol = c_base + tid % fx.Int32(BLOCK_C)
                    zrow = tid // fx.Int32(BLOCK_C)
                    for zp in range_constexpr(BLOCK_R // z_rows):
                        r_ = zrow0 + zrow + fx.Int32(zp * z_rows)
                        ok = (r_ < out_m) & (zcol < out_n)
                        off = arith.select(ok, r_ * out_n + zcol, zoob)
                        buffer_ops.buffer_store(zero_b, zo_rsrc, off)

        @flyc.jit
        def launch_fwd(
            A,
            B,
            OUT,
            A_scale,
            B_scale,
            OFFS,
            a_slabs: fx.Int32,
            n_blocks: fx.Int32,
            n_c_tiles: fx.Int32,
            out_m: fx.Int32,
            out_n: fx.Int32,
            stream: fx.Stream,
        ):
            kernel_fwd(
                A,
                B,
                OUT,
                A_scale,
                B_scale,
                OFFS,
                a_slabs,
                n_c_tiles,
                out_m,
                out_n,
                n_blocks,
                value_attrs={
                    "rocdl.waves_per_eu": _WAVES_PER_EU,
                    "rocdl.flat_work_group_size": "256,256",
                    # gpu.func attrs survive to llvm.func, and the
                    # `passthrough` list becomes raw LLVM function
                    # attributes. Attached only off the shipping
                    # setting so waves_per_eu=1 compiles byte-for-byte
                    # as it always has.
                    **(
                        {"passthrough": [["amdgpu-agpr-alloc", str(AGPR_NEED)]]}
                        if _WAVES_PER_EU != 1
                        else {}
                    ),
                },
            ).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)

        return launch_fwd

    @functools.lru_cache(maxsize=None)
    def cached_launch(
        K: int,
        N: int,
        E: int,
        BLOCK_C: int,
        BLOCK_R: int = 256,
        SWZ_XCD: int = 0,
        SWZ_G: int = 1,
        ZTAIL: bool = False,
        PRESHUF: bool = False,
        GUARD: bool = True,
    ):
        """Build the launcher. GUARD is part of the key -- see `launch_for`.

        NOTE: this does NOT trace. FlyDSL traces lazily, on the first call
        through `fast_launch`, so a tracer error surfaces at LAUNCH time and
        cannot be caught here. That is what `launch_for` is for.
        """
        return _compile(
            K,
            N,
            E,
            BLOCK_C,
            BLOCK_R,
            preshuf=PRESHUF,
            GUARD=GUARD,
            SWZ_XCD=SWZ_XCD,
            SWZ_G=SWZ_G,
            ZTAIL=ZTAIL,
        )

    def launch_for(
        K: int,
        N: int,
        E: int,
        BLOCK_C: int,
        BLOCK_R: int,
        SWZ_XCD: int = 0,
        SWZ_G: int = 1,
        ZTAIL: bool = False,
        PRESHUF: bool = False,
    ):
        """The guarded launcher, unless this shape already failed to trace.

        The surplus-slot guard is worth 1.2-1.74x but does not trace on every
        shape: as a bare `if active:`, K=8192 N=256 BLOCK_R=128 raised "cannot
        evaluate dynamic 'Boolean' as Python bool", and only when it came after
        ~10 other kernels in the same process -- which is why the 24-shape sweep
        and the full test suite both missed it. A pre-guard A/B at the same
        position in the same sequence traced fine, so it is the guard.

        Routing the condition through `proceed` made every shape tried since
        trace, but "every shape tried" is exactly the guarantee that failed
        last time, so the floor stays. It has to live at the LAUNCH site: an
        untraceable shape would otherwise raise RuntimeError out of the op, and
        `_flydsl_grouped_mm_impl` only falls back to Triton on
        NotImplementedError and AssertionError -- so this would take down a
        training job on a shape that worked yesterday. GUARD=False is verified
        bitwise-identical to GUARD=True; the floor costs throughput, nothing
        else.
        """
        key = (K, N, E, BLOCK_C, BLOCK_R, SWZ_XCD, SWZ_G, ZTAIL, PRESHUF)
        return cached_launch(
            K,
            N,
            E,
            BLOCK_C,
            BLOCK_R,
            SWZ_XCD,
            SWZ_G,
            ZTAIL,
            PRESHUF,
            GUARD=key not in _guard_broken,
        )

    def launch_guarded(launch, args, key):
        """Run `launch`, and on a tracer error retry once unguarded, forever."""
        try:
            return fast_launch(launch, *args)
        except Exception as e:
            if key in _guard_broken:
                raise
            _guard_broken.add(key)
            _warn_guard_fallback(*key, e)
            return fast_launch(cached_launch(*key, GUARD=False), *args)

else:

    def cached_launch(*_a, **_k):
        raise ImportError("FlyDSL runtime not available")


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def grouped_mm(
    input_act,
    weight,
    input_act_scales,
    weight_scales,
    group_end_offsets,
    out_dtype=torch.bfloat16,
    block_c=None,
    out=None,
    block_r=None,
):
    """MXFP8 forward grouped GEMM over RAGGED token groups.

        out[group_g] = input_act[group_g] @ weight[g]^T

    Groups partition the token (output row) dim with device-resident, unequal
    sizes. Because the ragged axis is NOT the contraction, the whole pipelined
    K-walk carries over from the even-groups body untouched -- only the
    block->(group, tile) mapping and the epilogue mask change.

    input_act         (M, K)      fp8 e4m3, row-major
    weight            (E, N, K)   fp8 e4m3, row-major
    input_act_scales  (M, K//32)  e8m0 as uint8
    weight_scales     (E, N, K//32)
    group_end_offsets (E,) int32  cumulative group ends along M, ON DEVICE.
                                  Read by the block itself -- never synced here.
    block_c           column tile width; None picks it from N (see pick_block_c)
    out               optional (M, N) bf16 destination to write in place.

    Pass ``out`` when you already have the destination -- it skips both an
    allocation and the zero-fill. That matters: the zero-fill is a full memset of
    M*N, 805 MB at the e2e shape, measured at 0.117 ms against a 1.225 ms kernel,
    i.e. **9.5%** -- almost the entire cost of ragged support over the even body.

    When we allocate, we zero. Rows at and past ``group_end_offsets[-1]`` are
    written by no block (the dispatcher pads M past the last group), and handing
    back uninitialised memory there is the worse default. Callers matching ATen's
    grouped-mm contract can skip it: ATen's own reference leaves that tail
    untouched too, so it is not part of the result.
    """
    M, K = input_act.shape
    E, N, K2 = weight.shape
    assert K == K2, f"K mismatch: A={K}, B={K2}"
    assert input_act_scales.shape == (M, K // SCALE_BLOCK), (
        f"x scales {tuple(input_act_scales.shape)} != {(M, K // SCALE_BLOCK)}"
    )
    assert weight_scales.shape == (E, N, K // SCALE_BLOCK), (
        f"w scales {tuple(weight_scales.shape)} != {(E, N, K // SCALE_BLOCK)}"
    )
    assert group_end_offsets.numel() == E, (
        f"offsets {group_end_offsets.numel()} != E {E}"
    )

    _br, _bc = pick_tile(M, E, N)
    BLOCK_C = _bc if block_c is None else int(block_c)
    BLOCK_R_T = _br if block_r is None else int(block_r)
    n_c = ceildiv(N, BLOCK_C)

    if out is None:
        # `empty`, not `zeros`: the rows at and past the last group's end are
        # zeroed BY THE KERNEL, from the surplus row-tile slots it was launching
        # and discarding anyway (see ZTAIL). The memset this replaces was the
        # whole M x N plane -- 10.7% of the 671B `w2` forward GEMM -- against a
        # tail the dispatcher bounds at E * PAD_MULTIPLE rows.
        ztail = True
        out = torch.empty(M, N, dtype=torch.bfloat16, device=input_act.device)
    else:
        # A caller-supplied buffer keeps the documented contract: we write this
        # call's rows and touch nothing else, tail included.
        ztail = False
        assert out.shape == (M, N) and out.dtype == torch.bfloat16, (
            f"out {tuple(out.shape)}/{out.dtype} != {(M, N)}/torch.bfloat16"
        )
        assert out.is_contiguous(), "out must be contiguous"

    a = input_act.contiguous().view(torch.int8)
    a_sc = input_act_scales.contiguous().view(torch.uint8)
    b = weight.contiguous().view(torch.int8).view(-1)
    b_sc = weight_scales.contiguous().view(torch.uint8).view(-1)
    offs = group_end_offsets.to(torch.int32).contiguous()
    swz_xcd, swz_g = pick_swizzle(N, BLOCK_C)
    # ── Pre-shuffled scales ──
    # Both planes are rewritten once per call into the layout the MFMA fragment
    # wants, which buys the inner loop its ds_bpermute and byte unpack back and
    # halves its scale loads (PACK=4; see _scale_preshuffle.py). The pass is
    # ~1% of the GEMM. A half is 64 rows over 4 tiles by construction, so this
    # only exists at the (256,256) tile; anything else keeps the cooperative
    # load. $TORCHAO_MXFP8_FLYDSL_GEMM_SCALE_PRESHUFFLE=0 forces it off. It is a compile-key
    # bit, so both bodies coexist in one process for an A/B.
    preshuf = (
        os.environ.get("TORCHAO_MXFP8_FLYDSL_GEMM_SCALE_PRESHUFFLE", "1") != "0"
        and BLOCK_R_T == 256
        and BLOCK_C == 256
    )
    key = (K, N, E, BLOCK_C, BLOCK_R_T, swz_xcd, swz_g, ztail, preshuf)
    launch = launch_for(*key)
    stream = torch.cuda.current_stream()
    b_sp = _sp.preshuffle_b(b_sc, E, N, K, BLOCK_C) if preshuf else b_sc

    for r0, rows, offs_w in _row_windows(M, K, N, offs, BLOCK_R_T):
        n_slots = ceildiv(rows, BLOCK_R_T) + E
        # The supertile rasterization is a bijection only when the slot range is
        # a whole number of supertiles; the extra slots are surplus and exit at
        # the `active` guard before any global traffic.
        if swz_g > 1:
            n_slots = ceildiv(n_slots, swz_g) * swz_g
        # The A plane is pre-shuffled per WINDOW, on the narrowed slice, because
        # its slab grid counts from the window's own row 0 -- the same reason the
        # offsets are re-based per window.
        a_sc_w = a_sc.narrow(0, r0, rows)
        if preshuf:
            a_sp = _sp.preshuffle_a(a_sc_w, rows, K, E, BLOCK_R_T, offs_w)
            a_slabs = _sp.a_slabs_ub(rows, E, BLOCK_R_T)
        else:
            a_sp, a_slabs = a_sc_w.view(-1), 0
        launch_guarded(
            launch,
            (
                a.narrow(0, r0, rows).view(-1),
                b,
                out.narrow(0, r0, rows).view(-1),
                a_sp,
                b_sp,
                offs_w,
                a_slabs,
                n_slots * n_c,
                n_c,
                rows,
                N,
                stream,
            ),
            key,
        )
        # A window that fell back re-resolves, so later windows skip the retry.
        launch = launch_for(*key)
    return out if out_dtype == torch.bfloat16 else out.to(out_dtype)


def _row_windows(M, K, N, offs, block_r=BLOCK_R):
    """Split the token dim into windows no single launch can overflow int32 on.

    FlyDSL's CABI packs a tensor's shape entries as int32
    (``jit_argument._LayoutPlan``: ``"i" * len(shape)``), and every operand goes
    over as a 1-D view, so ``shape[0]`` IS the element count. Past 2**31 the pack
    raises ``struct.error`` from inside the dispatch -- which kills the training
    job outright rather than failing one op, and is not one of the exceptions
    ``_flydsl_grouped_mm_impl`` falls back on. Reached on a real DSV3-16B step at
    65k tokens/rank: M=1,108,768 x K=2048 = 2,270,756,864.

    Splitting is sound HERE, and only here, because the groups partition the
    OUTPUT ROWS rather than the contraction: each window owns a disjoint row
    range, so the windows never share a partial sum and need no accumulation.
    (The wgrad cannot do this -- there the groups partition the contraction.)

    Yields ``(row_start, n_rows, window_offsets)``. The offsets are rebased and
    clamped ON DEVICE -- a group that straddles a boundary ends at ``n_rows`` in
    one window and starts at 0 in the next, and groups wholly outside collapse to
    size 0. No host sync, which is the property this whole kernel is built around.
    """
    # Bound the widest M-dependent operand: A is (M, K), out is (M, N), and A's
    # scales are (M, K//32).
    rows_max = (2**31 - 1) // max(K, N)
    # Align down to the row-tile size so a window boundary never splits a tile;
    # the tiling inside each window is then identical to the unsplit case.
    rows_max = (rows_max // block_r) * block_r
    if rows_max < block_r:
        raise NotImplementedError(
            f"a single row tile of {block_r} rows already exceeds the int32 "
            f"operand limit at K={K} N={N}"
        )

    if M <= rows_max:
        yield 0, M, offs  # unsplit: byte-for-byte the previous path
        return
    for r0 in range(0, M, rows_max):
        rows = min(rows_max, M - r0)
        yield r0, rows, (offs - r0).clamp_(0, rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

#: gfx950 only: the body is built on `v_mfma_scale_f32_16x16x128_f8f6f4`, the
#: scaled MFMA that takes the E8M0 block scales as operands. MI300 (gfx942) has
#: no such instruction, so unlike the FlyDSL quantization kernels alongside this
#: one, MI300 is not a target.
_mxfp8_flydsl_grouped_mm_available = (
    torch.cuda.is_available() and is_MI350() and _flydsl_runtime_available()
)

#: The contraction is walked in steps of BLOCK_M=128 elements, and the software
#: pipeline needs 2 prologue steps plus 2 tail steps before a steady state
#: exists, so K must be a multiple of 128 and at least 4 steps deep.
_MIN_K = 4 * BLOCK_M


def _check_flydsl_grouped_mm_unsupported_params(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
) -> None:
    """Reject what the kernel cannot do, before it can produce a wrong answer.

    M and N are unconstrained -- the row and column tiles are masked, so a
    partial tile at either edge is handled. Only K is constrained, by the
    pipeline depth.
    """
    if not _flydsl_runtime_available():
        missing = ", ".join(_missing_flydsl_runtime_packages())
        raise NotImplementedError(
            f"mxfp8_grouped_mm_flydsl requires the FlyDSL runtime; missing: {missing}. "
            "Install it with `pip install flydsl` (https://github.com/ROCm/FlyDSL)."
        )
    if not _mxfp8_flydsl_grouped_mm_available:
        raise NotImplementedError(
            "mxfp8_grouped_mm_flydsl requires an AMD MI350-series GPU (gfx950): "
            "it is built on the scaled MFMA instruction, which gfx942 does not have."
        )
    if A_data.ndim != 2:
        raise NotImplementedError(f"A_data must be 2D (M, K), got {A_data.shape}")
    if B_data.ndim != 3:
        raise NotImplementedError(f"B_data must be 3D (E, N, K), got {B_data.shape}")
    for name, t in (("A_data", A_data), ("B_data", B_data)):
        if t.dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"{name} must be float8_e4m3fn, got {t.dtype}")
    for name, t in (("A_scale", A_scale), ("B_scale", B_scale)):
        if t.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise NotImplementedError(
                f"{name} must hold E8M0 bytes (uint8 or float8_e8m0fnu), got {t.dtype}"
            )
    K = A_data.shape[-1]
    if K % BLOCK_M != 0 or K < _MIN_K:
        raise NotImplementedError(
            f"K must be a multiple of {BLOCK_M} and at least {_MIN_K}, got K={K}"
        )
    if offs.numel() != B_data.shape[0]:
        raise NotImplementedError(
            f"offs has {offs.numel()} entries but B_data has {B_data.shape[0]} groups"
        )
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(f"out_dtype must be bf16 or fp32, got {out_dtype}")


def mxfp8_grouped_mm_flydsl(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MXFP8 grouped GEMM on AMD gfx950: ``out[g] = A[group_g] @ B[g].T``.

    Serves both the MoE forward and the dgrad; for the dgrad, pass grad_output
    as A and the untransposed expert weight as B.

    Args:
        A_data: ``(M, K)`` float8_e4m3fn activations, row-major.
        A_scale: ``(M, K // 32)`` E8M0 block scales, row-major.
        B_data: ``(E, N, K)`` float8_e4m3fn expert weights, row-major.
        B_scale: ``(E, N, K // 32)`` E8M0 block scales.
        offs: ``(E,)`` int32 cumulative group END offsets along M. Group sizes
            may differ and may be zero. Read ON DEVICE by the kernel -- this
            function never syncs them to the host.
        out_dtype: ``torch.bfloat16`` (default) or ``torch.float32``.
        out: optional ``(M, N)`` bf16 destination written in place. Passing it
            skips an allocation; the kernel then writes only the rows this call
            owns and leaves the tail past ``offs[-1]`` untouched. When we
            allocate, that tail is zeroed.

    Returns:
        ``(M, N)`` tensor in ``out_dtype``.

    Raises:
        NotImplementedError: for a shape or dtype the kernel does not support
            (see ``_check_flydsl_grouped_mm_unsupported_params``).
    """
    _check_flydsl_grouped_mm_unsupported_params(
        A_data, A_scale, B_data, B_scale, offs, out_dtype
    )
    return grouped_mm(
        A_data,
        B_data,
        A_scale.view(torch.uint8),
        B_scale.view(torch.uint8),
        offs,
        out_dtype=out_dtype,
        out=out,
    )
