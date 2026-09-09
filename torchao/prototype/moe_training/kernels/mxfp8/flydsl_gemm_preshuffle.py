# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Pre-shuffle the forward/dgrad E8M0 scale planes into the layout lanes read.

Same idea, same payoff and the same layout as the wgrad's
``_ops/mxwgrad/_scale_preshuffle.py`` -- read that file's header first; it
explains why a lane cannot address its own e8m0 byte in the raw plane and what
``op_sel`` buys once it can. What differs here is only the SLAB GRID, and it
differs because this kernel's two operands are indexed differently:

* **A (activations)** is ``(M, K/32)`` and a block's first row is
  ``m_start + r_base``. ``r_base`` is a multiple of BLOCK_R, but ``m_start`` is a
  group boundary and the dispatcher only pads those to ``AMD_MXFP8_PAD_MULTIPLE``
  -- 32 in the high-precision-wgrad config. A 32-aligned start would put a
  block's 64 rows across two slabs of a plane-aligned grid, which no single
  ``buffer_load`` can serve. So A's slabs are **per group**: group g owns
  ``ceildiv(m_g, BLOCK_R) * (BLOCK_R/64)`` of them starting at a prefix ``pre_g``,
  and the block's slab is ``pre_g + (r_base + wave*64)/64`` -- exact for any group
  start. The pre-pass finds the owner of each slab with the same O(E) scan the
  main kernel already runs. (the reference implementation's grouped pre-shuffle does this too,
  for the same reason.)

* **B (weights)** is ``(E, N, K/32)`` with E, N and BLOCK_C all compile-time, so
  its slabs are **per expert** at a constexpr stride, no scan.

Both grids deliberately allocate `ceildiv(rows, BLOCK) * (BLOCK/64)` slabs rather
than `ceildiv(rows, 64)`: a block's last row tile runs past the group/expert when
the row count is not a whole number of tiles, and the slab index has to stay
exact out there. The surplus rows read 0 (an e8m0 0x00 = 2**-127) and the
epilogue's row/column masks drop their output, exactly as on the raw path.

K is a compile-time constant in this kernel, so the packed-K stride folds into
the IR and the pre-pass takes no runtime shape argument at all.
"""

# NOTE: no `from __future__ import annotations` -- fx.struct resolves the LDS
# schema's annotations at class-creation time and a string one is not Storable.

import functools

import torch

from .flydsl_launch import fast_launch
from .flydsl_utils import _flydsl_runtime_available

#: e8m0 bytes packed per output dword. 4, not the wgrad's 2: this kernel's
#: K-walk is FULLY unrolled, so the K-step index is a Python int and `op_sel =
#: k % PACK` is compile-time at any packing -- the wgrad is capped at 2 only
#: because its `scf.for` body covers two K-steps. At 4 a chunk serves four
#: K-steps, which HALVES the scale load count against the cooperative path at
#: identical dword volume, and that load count is worth 9% of the kernel here.
PACK = 4

#: K-steps (i32 columns of the raw plane) staged in LDS per workgroup. This is
#: the pass's READ granularity: consecutive threads walk KT columns of one row,
#: so a wave issues 64*4/KT separate segments of KT*4 bytes each. At 16 that is
#: 64-byte segments, half of what HBM wants; 32 makes them 128.
KT = 32
BLK = 256
ROWS_PER_GRP = 64
SCALE_BLOCK = 32


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def a_slabs_ub(M: int, E: int, block_r: int) -> int:
    """Upper bound on A's per-group slabs: the host cannot see the group sizes,
    so it launches for the worst case, the same way the GEMM over-provisions row
    tiles. One group can waste at most one row tile's worth."""
    return (ceildiv(M, block_r) + E) * (block_r // ROWS_PER_GRP)


def b_slabs_per_expert(N: int, block_c: int) -> int:
    return ceildiv(N, block_c) * (block_c // ROWS_PER_GRP)


def sp_elems(n_slabs: int, k128: int) -> int:
    return n_slabs * ceildiv(k128, PACK) * 64 * 4


if _flydsl_runtime_available():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, range_constexpr
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec

    from . import flydsl_buffer_ops as buffer_ops
    from .flydsl_gemm_utils import wait_barrier

    # Sending a store past the descriptor is how a surplus slot is dropped: the
    # index it would otherwise write is INSIDE the allocation (it aliases the
    # next slab), so num_records does not catch it on its own.
    _DROP = 0x7FFFFFF0

    def _lds_slot(tile, idx):
        return fx.make_view(
            fx.add_offset(tile.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)
        )

    def _repack(
        tile, rin, rout, tid, base_row, out_slab, k0, K128, K128p, row_lo, row_hi, valid
    ):
        """Stage 64 rows x KT i32 through LDS, emit them lane-major and packed.

        `base_row` is the plane row of this slab's row 0; `row_lo`/`row_hi` bound
        which of the 64 rows are real (past them, e8m0 0x00); `valid` says whether
        this slab exists at all.
        """
        TILE = ROWS_PER_GRP * KT
        for i in range_constexpr(TILE // BLK):
            idx = tid + fx.Int32(i * BLK)
            rr = idx // fx.Int32(KT)
            kk = idx % fx.Int32(KT)
            gk = k0 + kk
            row = base_row + rr
            dw = buffer_ops.buffer_load(
                rin, row * fx.Int32(K128) + gk, vec_width=1, dtype=T.i32
            )
            # A column over-run lands on the NEXT row's scales (one flat
            # allocation), and a row over-run on the next group's, so neither is
            # caught by the descriptor. Both are zeroed here.
            ok = (gk < fx.Int32(K128)) & (row >= row_lo) & (row < row_hi)
            dw = arith.select(ok, ArithValue(dw), fx.Int32(0))
            _lds_slot(tile, idx).store(Vec.from_elements([dw], fx.Int32))

        # vmcnt(0) for the staged loads; wait_barrier's lgkmcnt(0) covers the
        # LDS writes themselves.
        wait_barrier(0)

        NOUT = (KT // PACK) * 64
        for j in range_constexpr(NOUT // BLK):
            ol = tid + fx.Int32(j * BLK)
            kkp = ol // fx.Int32(64)
            lane = ol % fx.Int32(64)
            r = lane % fx.Int32(16)
            sh = (lane // fx.Int32(16)) * fx.Int32(8)
            elems = []
            for t in range_constexpr(4):
                packed = fx.Int32(0)
                for b in range_constexpr(PACK):
                    so = (
                        (fx.Int32(t * 16) + r) * fx.Int32(KT)
                        + kkp * fx.Int32(PACK)
                        + fx.Int32(b)
                    )
                    v = Vec(_lds_slot(tile, so).load())
                    byte = (ArithValue(fx.Int32(v[0])) >> sh) & fx.Int32(0xFF)
                    packed = packed | (byte << fx.Int32(b * 8))
                elems.append(packed)
            gkp = k0 // fx.Int32(PACK) + kkp
            off = ((out_slab * fx.Int32(K128p) + gkp) * fx.Int32(64) + lane) * fx.Int32(
                4
            )
            off = arith.select(valid & (gkp < fx.Int32(K128p)), off, fx.Int32(_DROP))
            buffer_ops.buffer_store(Vec.from_elements(elems, fx.Int32), rout, off)

    @fx.struct
    class _Smem:
        tile: fx.Array[fx.Int32, ROWS_PER_GRP * KT, 16]

    def _compile_a(K: int, E: int, block_r: int):
        K128 = K // 128
        K128p = ceildiv(K128, PACK)
        n_kt = ceildiv(K128, KT)
        SPT = block_r // ROWS_PER_GRP  # slabs per row tile

        @flyc.kernel(known_block_size=[BLK, 1, 1])
        def kern(
            SRC: fx.Tensor,
            DST: fx.Tensor,
            OFFS: fx.Tensor,
            out_m: fx.Int32,
            n_slabs: fx.Int32,
        ):
            tid = ArithValue(fx.thread_idx.x)
            bid = ArithValue(fx.block_idx.x)
            sl = bid // fx.Int32(n_kt)
            k0 = (bid % fx.Int32(n_kt)) * fx.Int32(KT)

            offs_rsrc = buffer_ops.create_buffer_resource(
                OFFS, max_size=False, num_records_bytes=E * 4
            )
            ends = [
                ArithValue(
                    buffer_ops.buffer_load(
                        offs_rsrc, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
                    )
                )
                for i in range(E)
            ]
            starts = [ArithValue(fx.Int32(0))] + ends[:-1]

            # The same O(E) scan the GEMM runs, over SLABS instead of row tiles.
            base_row = ArithValue(fx.Int32(0))
            row_lo = ArithValue(fx.Int32(0))
            row_hi = ArithValue(fx.Int32(0))
            valid = fx.Int32(0) > fx.Int32(0)
            cum = ArithValue(fx.Int32(0))
            for i in range_constexpr(E):
                ns = (
                    (ends[i] - starts[i] + fx.Int32(block_r - 1)) // fx.Int32(block_r)
                ) * fx.Int32(SPT)
                hit = (sl >= cum) & (sl < cum + ns)
                base_row = arith.select(
                    hit, starts[i] + (sl - cum) * fx.Int32(ROWS_PER_GRP), base_row
                )
                row_lo = arith.select(hit, starts[i], row_lo)
                row_hi = arith.select(hit, ends[i], row_hi)
                valid = valid | hit
                cum = cum + ns

            tile = fx.SharedAllocator().allocate(_Smem).peek().tile
            src_bytes = arith.index_cast(
                T.i64, arith.index_cast(T.index, out_m * fx.Int32(K128 * 4))
            )
            dst_bytes = arith.index_cast(
                T.i64, arith.index_cast(T.index, n_slabs * fx.Int32(K128p * 64 * 4 * 4))
            )
            rin = buffer_ops.create_buffer_resource(
                SRC, max_size=False, num_records_bytes=src_bytes
            )
            rout = buffer_ops.create_buffer_resource(
                DST, max_size=False, num_records_bytes=dst_bytes
            )
            _repack(
                tile,
                rin,
                rout,
                tid,
                base_row,
                sl,
                k0,
                K128,
                K128p,
                row_lo,
                row_hi,
                valid,
            )

        @flyc.jit
        def launch(
            SRC,
            DST,
            OFFS,
            n_blocks: fx.Int32,
            out_m: fx.Int32,
            n_slabs: fx.Int32,
            stream: fx.Stream,
        ):
            kern(SRC, DST, OFFS, out_m, n_slabs).launch(
                grid=(n_blocks, 1, 1), block=(BLK, 1, 1), stream=stream
            )

        return launch

    def _compile_b(K: int, E: int, N: int, block_c: int):
        K128 = K // 128
        K128p = ceildiv(K128, PACK)
        n_kt = ceildiv(K128, KT)
        SPE = b_slabs_per_expert(N, block_c)  # slabs per expert, constexpr
        NSLAB = E * SPE

        @flyc.kernel(known_block_size=[BLK, 1, 1])
        def kern(SRC: fx.Tensor, DST: fx.Tensor):
            tid = ArithValue(fx.thread_idx.x)
            bid = ArithValue(fx.block_idx.x)
            sl = bid // fx.Int32(n_kt)
            k0 = (bid % fx.Int32(n_kt)) * fx.Int32(KT)
            g = sl // fx.Int32(SPE)
            loc = sl % fx.Int32(SPE)

            tile = fx.SharedAllocator().allocate(_Smem).peek().tile
            rin = buffer_ops.create_buffer_resource(
                SRC, max_size=False, num_records_bytes=E * N * K128 * 4
            )
            rout = buffer_ops.create_buffer_resource(
                DST, max_size=False, num_records_bytes=NSLAB * K128p * 64 * 4 * 4
            )
            base = g * fx.Int32(N) + loc * fx.Int32(ROWS_PER_GRP)
            _repack(
                tile,
                rin,
                rout,
                tid,
                base,
                sl,
                k0,
                K128,
                K128p,
                g * fx.Int32(N),
                (g + fx.Int32(1)) * fx.Int32(N),
                fx.Int32(1) > fx.Int32(0),
            )

        @flyc.jit
        def launch(SRC, DST, n_blocks: fx.Int32, stream: fx.Stream):
            kern(SRC, DST).launch(
                grid=(n_blocks, 1, 1), block=(BLK, 1, 1), stream=stream
            )

        return launch

    @functools.lru_cache(maxsize=None)
    def cached_a(K: int, E: int, block_r: int):
        return _compile_a(K, E, block_r)

    @functools.lru_cache(maxsize=None)
    def cached_b(K: int, E: int, N: int, block_c: int):
        return _compile_b(K, E, N, block_c)

else:  # pragma: no cover - no runtime

    def cached_a(*_a):
        raise RuntimeError("fwdgemm._scale_preshuffle is unavailable")

    cached_b = cached_a


_WS: dict = {}


def _workspace(tag, n_slabs: int, k128: int, device) -> "torch.Tensor":
    """Grow-only, one per (tag, device, stream). `n_slabs` is deliberately out of
    the key: it tracks M, which changes every step under real routing, and a
    keyed cache would allocate a plane per distinct step size and free none."""
    key = (tag, device.index, torch.cuda.current_stream(device).cuda_stream)
    need = sp_elems(n_slabs, k128)
    ws = _WS.get(key)
    if ws is None or ws.numel() < need:
        ws = torch.empty(need, dtype=torch.int32, device=device)
        _WS[key] = ws
    return ws


def preshuffle_a(a_sc, M: int, K: int, E: int, block_r: int, offs):
    n_slabs = a_slabs_ub(M, E, block_r)
    k128 = K // 128
    dst = _workspace(("a", K, E, block_r), n_slabs, k128, a_sc.device)
    src = a_sc.contiguous().view(torch.int32).reshape(-1)
    fast_launch(
        cached_a(K, E, block_r),
        src,
        dst,
        offs,
        n_slabs * ceildiv(k128, KT),
        M,
        n_slabs,
        torch.cuda.current_stream(),
    )
    return dst


def preshuffle_b(b_sc, E: int, N: int, K: int, block_c: int):
    n_slabs = E * b_slabs_per_expert(N, block_c)
    k128 = K // 128
    dst = _workspace(("b", K, E, N, block_c), n_slabs, k128, b_sc.device)
    src = b_sc.contiguous().view(torch.int32).reshape(-1)
    fast_launch(
        cached_b(K, E, N, block_c),
        src,
        dst,
        n_slabs * ceildiv(k128, KT),
        torch.cuda.current_stream(),
    )
    return dst
