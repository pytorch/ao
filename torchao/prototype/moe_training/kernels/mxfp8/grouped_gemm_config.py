# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen configuration for the shared MXFP8 grouped blockscaled GEMM core.

One tiling, one pipeline shape, one warp assignment for all three kernels in the
family (FC1 GEMM+SwiGLU, FC2 dgrad+dSwiGLU, grouped wgrad). Kernels differ only
in their epilogue and in ``epi_n_acc``; nothing else here is a per-kernel knob.

Why each value is what it is:

``cta_group = ONE`` / ``cluster_shape_mn = (1, 1)``
    Every per-expert row count is a multiple of 128 and ``cta_tile_m`` is 128, so
    an M tile never straddles an expert boundary. 2CTA forces a 256-row cluster
    tile, which reintroduces partial tiles for ``m[g] == 128 (mod 256)`` and
    would need hand-predication in every quantized store and every scale byte.
    With one CTA per MMA there is also nothing to multicast, so the cluster is
    trivial and no multicast masks or cluster launch barriers exist.

``mma_tiler = (128, 128, 128)``
    M=128 is mandatory for ``CtaGroup.ONE``. K=128 is the smallest multiple of
    ``sf_vec_size * 4`` and makes one K tile exactly one scale-factor atom. N=128
    keeps TMEM at 160 of 512 columns and avoids both the N=64 SFB column-shift
    and the N=256 overlapping-accumulator machinery.

``num_acc_stage = 1``
    One CTA produces exactly one output tile (the grid is data-independent and
    there is no persistent tile scheduler), so there is no second accumulator to
    overlap with.

``threads = 256``
    Warp 0 loads (TMA), warp 1 issues the MMA, warps 4-7 run the epilogue, warps
    2-3 idle. 128 epilogue threads is what makes ``tmem_warp_shape_mn = (4, 1)``
    and the thread-owns-one-row identity hold; see :data:`T2R_PARTITION_DOC`.

``epi_tile = (128, epi_n_acc)``
    ``epi_tile_M == cta_tile_M == 128`` is mandatory: every gate/up pairing and
    every scale index below assumes a single epilogue tile along M.

CTA_N = 256 is a legal alternative (TMEM 256+16+32 = 304 <= 512 with one
accumulator stage) and changes only the rowwise scale store width, but it halves
``num_ab_stage``; nothing structural depends on the choice.

WARNING -- ``cta_tile_k`` is pinned at 128 and raising it is not a free tuning
knob. The grouped wgrad kernel selects an expert's K range with an integer K-tile
index base rather than a per-expert TMA descriptor, which is exact only because
``token_offset`` (a multiple of 128) is a multiple of ``cta_tile_k``. At
``cta_tile_k = 256`` a group boundary can land mid-tile, and the kernel would
need per-expert descriptors or explicit K-tail predication brought back in. The
constant-512-byte ``Rest_K`` stride that makes the index base correct is a
property of the 128-element scale-factor granule, not of the tile size, so it
does not rescue a larger tile either.
"""

import enum
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "RaggedAxis",
    "GroupedGemmConfig",
    "SWIGLU_FWD_CONFIG",
    "DSWIGLU_BWD_CONFIG",
    "WGRAD_CONFIG",
    "SMEM_CAPACITY_BYTES",
    "TMEM_TOTAL_COLS",
    "is_supported",
    "check_supported",
    "T2R_PARTITION_DOC",
    "EPILOGUE_PROTOCOL_DOC",
]

# sm_100 usable dynamic shared memory per CTA, (228 - 1) KiB.
SMEM_CAPACITY_BYTES = 232448
# The TMEM allocator requires a power-of-two multiple of 32 columns; shared
# memory already pins us to one CTA per SM, so allocating the whole array costs
# nothing.
TMEM_TOTAL_COLS = 512
# Row-count granularity every expert group, and the allocation itself, respect.
GROUP_ALIGNMENT = 128
# MXFP8 scaling block: 32 values share one E8M0 scale.
SF_VEC_SIZE = 32


class RaggedAxis(enum.Enum):
    """Which GEMM axis the per-expert offsets partition.

    ``M`` is kernels A and B: the ragged axis is the token axis, the grid is
    ``(R // 128, N // CTA_N, 1)``, and the expert is looked up per CTA from the
    absolute row base. ``K`` is kernel C: the ragged axis is the contraction, the
    grid is ``(N // 128, K // 128, G)``, and only the K-loop trip count is
    data-dependent.
    """

    M = 0
    K = 1


@dataclass(frozen=True)
class GroupedGemmConfig:
    """Constexpr configuration of the shared grouped blockscaled GEMM core.

    Every field is a trace-time constant. No kernel body may hard-code any of
    these numbers; read them from the config so a retune cannot silently
    desynchronize the mainloop from an epilogue.
    """

    # Per-kernel: accumulator columns handed to the epilogue per subtile.
    # 64 for the FC1 SwiGLU forward (one output column consumes an adjacent
    # gate/up accumulator pair, so 64 accumulator columns are 32 output columns
    # == exactly one rowwise 1x32 block per thread), 32 for the dgrad backward
    # and for wgrad.
    epi_n_acc: int
    # Which axis the offsets partition.
    ragged_axis: RaggedAxis

    # --- frozen for every kernel in the family -------------------------------
    cta_tile_m: int = 128
    cta_tile_n: int = 128
    cta_tile_k: int = 128
    num_ab_stage: int = 6
    num_acc_stage: int = 1
    cluster_shape_mn: Tuple[int, int] = (1, 1)
    sf_vec_size: int = SF_VEC_SIZE
    threads: int = 256
    tma_warp_id: int = 0
    mma_warp_id: int = 1
    epilogue_warp_ids: Tuple[int, ...] = (4, 5, 6, 7)
    # Named barrier ids. 0 is left free for the DSL's own use.
    epilogue_sync_barrier_id: int = 1
    tmem_alloc_barrier_id: int = 2

    def __post_init__(self):
        if self.cta_tile_m != 128:
            raise ValueError(
                "cta_tile_m is pinned at 128: CtaGroup.ONE requires it and the "
                "no-partial-M-tile argument depends on it"
            )
        if self.cta_tile_k != 128:
            raise ValueError(
                "cta_tile_k is pinned at 128; see this module's docstring for why "
                "raising it reintroduces per-expert descriptors"
            )
        if self.cta_tile_n % 128 != 0:
            # SFB's MN extent is round_up(N, 128). Only when that equals N is the
            # SFB tiled MMA identical to the data one, which is what lets the core
            # build a single tiled MMA and skip the N=64 TMEM column-shift path.
            raise ValueError(
                f"cta_tile_n must be a multiple of 128, got {self.cta_tile_n}; "
                "a smaller N needs a separate SFB tiled MMA and a TMEM column shift"
            )
        if self.cta_tile_n % self.epi_n_acc != 0:
            raise ValueError(
                f"cta_tile_n ({self.cta_tile_n}) must be a multiple of epi_n_acc "
                f"({self.epi_n_acc})"
            )
        if self.cta_tile_m % (4 * SF_VEC_SIZE) != 0:
            # 4 epilogue warps x 32 rows: a columnwise 32x1 block must never be
            # split across warps.
            raise ValueError(
                "cta_tile_m must be a multiple of 128 for the 4-warp epilogue"
            )
        if len(self.epilogue_warp_ids) * 32 != self.cta_tile_m:
            raise ValueError(
                "the epilogue must have exactly one thread per row of the CTA tile"
            )
        if self.epilogue_warp_ids[0] % 4 != 0:
            # tcgen05.ld selects its TMEM datapath sub-partition from the
            # PHYSICAL warp id, so the epilogue must start on an aligned warp
            # quad. A misaligned block still launches and still returns
            # plausible numbers -- every 128-row tile comes back with its four
            # 32-row datapath groups rotated -- and no shape, byte or NaN check
            # detects a pure row permutation. Reject it here.
            raise ValueError(
                f"epilogue_warp_ids must start on an aligned warp quad, got "
                f"{self.epilogue_warp_ids}: warp {self.epilogue_warp_ids[0]} is "
                f"{self.epilogue_warp_ids[0] % 4} past a multiple of 4; tcgen05.ld "
                "would silently permute the 32-row groups of every tile"
            )
        # The kernel selects the epilogue with `warp_idx >= epilogue_warp_ids[0]`,
        # so they must be the last contiguous block of warps in the CTA.
        if tuple(self.epilogue_warp_ids) != tuple(
            range(self.threads // 32 - len(self.epilogue_warp_ids), self.threads // 32)
        ):
            raise ValueError(
                f"epilogue_warp_ids {self.epilogue_warp_ids} must be the last "
                f"{len(self.epilogue_warp_ids)} warps of the {self.threads // 32} "
                "in the CTA"
            )
        if self.tma_warp_id in self.epilogue_warp_ids or (
            self.mma_warp_id in self.epilogue_warp_ids
        ):
            raise ValueError("the TMA and MMA warps must not be epilogue warps")

    # --- derived, all trace-time ---------------------------------------------

    @property
    def mma_tiler_mnk(self) -> Tuple[int, int, int]:
        return (self.cta_tile_m, self.cta_tile_n, self.cta_tile_k)

    @property
    def cta_tile_shape_mnk(self) -> Tuple[int, int, int]:
        return (self.cta_tile_m, self.cta_tile_n, self.cta_tile_k)

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return (self.cta_tile_m, self.epi_n_acc)

    @property
    def num_epi_subtiles(self) -> int:
        return self.cta_tile_n // self.epi_n_acc

    @property
    def num_epilogue_threads(self) -> int:
        return 32 * len(self.epilogue_warp_ids)

    @property
    def first_epilogue_thread(self) -> int:
        return 32 * self.epilogue_warp_ids[0]

    # --- shared memory budget ------------------------------------------------

    @property
    def ab_stage_bytes(self) -> int:
        """Bytes of one mainloop stage: A and B tiles plus both scale atoms.

        At (128, 128, 128) E4M3 that is 16384 + 16384 + 512 + 512 = 33792, which
        is also the exact ``tx_count`` the TMA pipeline barrier must expect.
        """
        a = self.cta_tile_m * self.cta_tile_k  # E4M3, 1 byte per element
        b = self.cta_tile_n * self.cta_tile_k
        # One E8M0 byte per 32 contracted elements per MN row, i.e. exactly one
        # 128x4 blocked tile (512 B) per operand at the frozen shape. SFB's MN
        # extent is round_up(N, 128).
        sfa = self.cta_tile_m * (self.cta_tile_k // SF_VEC_SIZE)
        sfb = max(self.cta_tile_n, 128) * (self.cta_tile_k // SF_VEC_SIZE)
        return a + b + sfa + sfb

    @property
    def mbarrier_bytes(self) -> int:
        """AB pipeline (full+empty) and accumulator handoff (full+empty) mbarriers."""
        return 8 * (2 * self.num_ab_stage + 2 * self.num_acc_stage)

    def smem_bytes(self, epilogue_smem_bytes: int = 0) -> int:
        return (
            self.num_ab_stage * self.ab_stage_bytes
            + self.mbarrier_bytes
            + epilogue_smem_bytes
            # tmem holding buffer plus struct alignment slack
            + 256
        )

    def max_ab_stages(self, epilogue_smem_bytes: int = 0) -> int:
        """Stages that fit alongside the epilogue's shared-memory request."""
        fixed = self.mbarrier_bytes + epilogue_smem_bytes + 256
        return (SMEM_CAPACITY_BYTES - fixed) // self.ab_stage_bytes

    # --- tensor memory budget ------------------------------------------------

    @property
    def acc_tmem_cols(self) -> int:
        return self.cta_tile_n * self.num_acc_stage

    @property
    def sfa_tmem_cols(self) -> int:
        return (self.cta_tile_m // SF_VEC_SIZE) * 4

    @property
    def sfb_tmem_cols(self) -> int:
        # SFB's MN extent is round_up(N, 128); at N=128 that is N itself.
        return (max(self.cta_tile_n, 128) // SF_VEC_SIZE) * 4

    @property
    def used_tmem_cols(self) -> int:
        """160 of 512 at the frozen shape."""
        return self.acc_tmem_cols + self.sfa_tmem_cols + self.sfb_tmem_cols


# Kernel A. EPI_N_ACC=64 is not negotiable: accumulator column 2f is gate_f and
# 2f+1 is up_f, so 64 accumulator columns are 32 output columns, exactly one
# rowwise 1x32 block per thread. 32 would give half a block and force a partial
# amax carried across subtiles.
SWIGLU_FWD_CONFIG = GroupedGemmConfig(epi_n_acc=64, ragged_axis=RaggedAxis.M)
# Kernel B. The accumulator N axis is F (one column per feature) and each column
# produces two interleaved output columns, so 32 accumulator columns are 64
# output columns = two rowwise 1x32 blocks per thread per subtile.
DSWIGLU_BWD_CONFIG = GroupedGemmConfig(epi_n_acc=32, ragged_axis=RaggedAxis.M)
# Kernel C. 32 FP32 accumulators are 32 contiguous BF16 outputs = 64 contiguous
# bytes per thread, 64-byte aligned.
WGRAD_CONFIG = GroupedGemmConfig(epi_n_acc=32, ragged_axis=RaggedAxis.K)


def is_supported(
    model_dim: int, hidden_dim: int, allocated_rows: int, num_groups: int
) -> bool:
    """The initial optimized support predicate, as a pure boolean.

    Mirrors the kernel contract's shape predicate and the host validators. The
    caller is expected to fall back to the unfused torchao path when this is
    false rather than launching -- feeding non-128-aligned groups to the blocked
    scale path produces a wrong-sized buffer and an unusable CUDA context, not an
    error.

    The per-expert row counts live in device memory and are not checkable here;
    they are asserted on device on every launch.
    """
    return (
        num_groups >= 1
        and model_dim > 0
        and hidden_dim > 0
        and allocated_rows > 0
        and model_dim % GROUP_ALIGNMENT == 0
        and hidden_dim % GROUP_ALIGNMENT == 0
        and allocated_rows % GROUP_ALIGNMENT == 0
    )


def check_supported(
    model_dim: int, hidden_dim: int, allocated_rows: int, num_groups: int
) -> None:
    """:func:`is_supported` with a message naming the offending value.

    Raises ``ValueError``, never ``assert``: ``python -O`` strips assertions and
    would reintroduce the silent-corruption path.
    """
    if num_groups < 1:
        raise ValueError(f"G must be at least 1, got {num_groups}")
    for name, value in (
        ("D", model_dim),
        ("F", hidden_dim),
        ("R", allocated_rows),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if value % GROUP_ALIGNMENT != 0:
            raise ValueError(
                f"{name} must be a multiple of {GROUP_ALIGNMENT}, got {value}"
            )


# ---------------------------------------------------------------------------
# The two frozen interfaces lanes 2 and 3 compile against.
# ---------------------------------------------------------------------------

T2R_PARTITION_DOC = """\
grouped_gemm_core.t2r_partition(tidx, tAcc, config) -> (tiled_copy_t2r,
tTR_tAcc, tTR_rAcc, tTR_cAcc)

  tidx   Int32 epilogue-local thread index in [0, 128). NOT the raw threadIdx.x;
         subtract config.first_epilogue_thread first.
  tAcc   the (MMA, MMA_M, MMA_N, ACC_STAGE) TMEM accumulator tensor.
  config a GroupedGemmConfig, which fixes epi_tile = (128, epi_n_acc).

Returns, with EPI_M == 1 always:

  tiled_copy_t2r  the tcgen05 TMEM->register tiled copy.
  tTR_tAcc        (T2R, T2R_M, T2R_N, EPI_M, EPI_N) TMEM source, sliced per
                  epilogue subtile s as tTR_tAcc[(None, None, None, 0, s)].
  tTR_rAcc        (T2R, T2R_M, T2R_N) FP32 register destination for one subtile.
  tTR_cAcc        (T2R, T2R_M, T2R_N, EPI_M, EPI_N) of (row, col) coordinates in
                  the 128 x cta_tile_n CTA tile, partitioned identically.

The copy atom is built with elem_ty_d = Float32 even though the real outputs are
E4M3. Passing an 8-bit d type steers get_tmem_load_op into the tmem_dp=16
layouts, which are shaped for a direct FP8 TMA store and are not what a
dual-quantization epilogue wants.

Structural consequence, measured on GB200 with a real TMEM accumulator (not just
from the copy atom's thread-value layout): thread t owns row t of the 128-row CTA
tile and all epi_n_acc accumulator columns of the subtile, contiguously, and the
raw register order is linear -- register v of thread t holds accumulator element
(t, v) of the subtile. Warp w therefore owns rows [32w, 32w+32) = exactly one
32-row MX block. So a rowwise 1x32 amax is intra-thread and a columnwise 32x1
amax is intra-warp, and a columnwise scale block is never split across warps or
CTAs.

Even so, derive every index -- the gate/up de-interleave and every scale
coordinate -- from tTR_cAcc rather than from a raw register number. The linear
order is a measured property of one copy atom at one epi_tile, not a guarantee,
and tTR_cAcc costs nothing because it folds at trace time.
"""

EPILOGUE_PROTOCOL_DOC = """\
An epilogue is a module-level function passed to the core as a Constexpr. Two
requirements, both load-bearing:

1. It must be decorated ``@cute.jit``. The DSL preprocessor only rewrites
   decorated functions, so an undecorated epilogue cannot use
   ``cutlass.range_constexpr`` (it raises "range_constexpr should be preprocessed
   by preprocessor") and, worse, a dynamic ``if`` in one would be evaluated as a
   Python truth test instead of becoming a predicated region.
2. It must be a module-level function object. The DSL keys its compile cache on
   function identity, so a lambda or a closure built per call recompiles every
   launch.

    @cute.jit
    def my_epilogue(
        tTR_rAcc,      # (T2R, T2R_M, T2R_N) FP32 register fragment, one subtile
        tTR_cAcc_s,    # (T2R, T2R_M, T2R_N) matching (row, col) coordinates in
                       # the 128 x cta_tile_n CTA tile
        tiled_copy_t2r,# for cute.make_tiled_copy_D / retile, if needed
        epi_tidx,      # Int32 in [0, 128); equals the CTA-tile row this thread owns
        subtile_idx,   # Constexpr int in [0, config.num_epi_subtiles)
        tile,          # TileCoords: see below
        epi_smem,      # cute.Pointer(Int32) to the requested scratch, or None
        out,           # the tuple of destination tensors the launcher passed
        cfg,           # Constexpr GroupedGemmConfig
    ) -> None

TileCoords fields, all Int32 and all CTA-uniform:

    tile_m     absolute M-tile index (kernels A/B: over [0, R/128))
    tile_n     absolute N-tile index
    expert     selected expert index. Meaningless on an inactive-tail tile, where
               the scan saturates at G-1; nothing may depend on it there, since
               that tile's output is defined to be zero.
    row_base   tile_m * 128
    col_base   tile_n * cta_tile_n
    k_cnt      mainloop trip count; 0 on a tail tile or a zero-token expert

The epilogue is called once per subtile, num_epi_subtiles times per CTA, always
by all 128 epilogue threads and always with k_cnt CTA-uniform.

Tail rule, stated precisely because the two halves pull in opposite directions:

  * NEVER predicate a STORE on the inactive tail. A tail tile arrives with a
    zeroed accumulator and the unmodified store path then emits exactly the zeros
    the contract requires (zero qdata bytes, zero E8M0 scale bytes). Skipping a
    store is how a destination element stops being written.

  * ALWAYS predicate an extra GMEM INPUT LOAD on `tile.k_cnt == 0`, substituting
    zeros. This is not optional. Kernel B reads the saved `z_bf16` in its
    epilogue, and rows [A, R) of that tensor are read-forbidden precisely because
    they may hold anything. Measured, feeding a tail `z` into the dSwiGLU with
    dh == 0 as the zeroed accumulator delivers it:
        z = 0                  -> dz 0x00000000                    correct
        z = NaN or +Inf        -> dz 0x7fff7fff  (dgate/dup = NaN)  WRONG
        z = uninit 0xDEADBEEF  -> dz 0x80008000  (dgate/dup = -0.0) WRONG
    A NaN tail makes that block's scale byte 0xFF and every qdata byte 0x7F;
    even benign garbage yields qdata 0x80 rather than 0x00. Both violate the
    read-forbidden and write-zero halves of the ragged-tail contract.

In short: the accumulator is already zeroed for you, so trust it and store
unconditionally; anything you load yourself must be gated on k_cnt.

No cross-subtile state: EPILOGUE is called once per subtile and every call gets
fresh registers. There is deliberately no way to carry a value from subtile s to
s+1, so rowwise scale bytes cannot be buffered across subtiles and emitted as one
wide store per CTA tile. DECIDED 2026-08-16: emit ONE scale store per subtile
(`rowwise_scale_flush(..., NUM_BYTES=1)`). That costs 2 stores instead of 1 for
Kernel A and 8 instead of 2 for Kernel B, and is verified bitwise clean. Store-count
reduction is a tuning-stage concern; buying it here would mean either
trace-time module-level state across the unrolled loop or staging through
epi_smem, both of which trade a correctness-critical interface for a few stores.

Shared-memory scratch: the epilogue declares its byte count to the launcher,
which passes back a 128-byte-aligned pointer of that size. The mainloop's stage
count is computed against that request, so it is not free -- but 9216 bytes (the
columnwise transpose staging) still leaves 6 stages.
"""
