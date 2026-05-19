"""RHT utility functions: Hadamard matrix construction, sign vector helpers, and Triton JIT helpers.

Provides get_wgrad_sign_vector, get_hadamard_matrix, get_rht_matrix, cast_to_fp4x2,
_compute_pid, and the NVFP4 quantization / scale-factor store helpers (formerly fp4_triton_ops).
"""

import functools
import math

import torch
from torch.utils._triton import has_triton

_TMA_WORKSPACES: dict = {}


def _device_key(device) -> str:
    """Normalize device to a canonical string key (e.g. 'cuda' and 'cuda:0' → 'cuda:0').

    torch.device('cuda') and torch.device('cuda:0') stringify differently but refer
    to the same physical device. Without normalization, callers using one form miss
    cache entries created by callers using the other, causing spurious re-allocations
    inside FakeTensor tracing mode (which produce FakeTensors instead of real tensors).
    """
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(d)


def _prewarm_rht_matrix(
    sign_vector: tuple[int, ...] | None,
    device: torch.device,
) -> None:
    # lru_cache keys distinguish positional/defaulted and keyword/defaulted calls.
    get_rht_matrix(sign_vector, device, torch.bfloat16, 16)
    get_rht_matrix(sign_vector=sign_vector, device=device, hadamard_dimension=16)


def prepare_for_cuda_graph(
    device,
    nbytes: int = 131072,
    *,
    sign_vectors: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    """Pre-allocate per-device persistent state required for torch.compile CUDA graphs.

    Must be called once per device before torch.compile to ensure allocations
    happen outside the CUDA graph pool. Subsequent calls return the cached TMA
    workspace (no new allocation). Kernels run sequentially in the same CUDA stream
    and safely alias the TMA buffer.

    Also pre-warms get_rht_matrix (lru_cache) to prevent pool-allocation errors
    during graph capture. Pass any explicit RHT sign vectors used by the graph
    through sign_vectors so those cache entries are allocated before capture.
    """
    key = _device_key(device)
    if key not in _TMA_WORKSPACES:
        _TMA_WORKSPACES[key] = torch.empty(nbytes, dtype=torch.uint8, device=device)
    # Pre-warm every call because tests and callers may clear get_rht_matrix's
    # cache after the workspace has already been initialized. Use
    # torch.device(key) ("cuda:N") because A.device always produces a fully
    # indexed device and lru_cache keys are compared by value.
    _dev = torch.device(key)
    _prewarm_rht_matrix(None, _dev)
    for sign_vector in sign_vectors or ():
        _prewarm_rht_matrix(tuple(sign_vector), _dev)
    return _TMA_WORKSPACES[key]


def get_wgrad_sign_vector(
    shape, device, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Generate a random {-1, 1} sign vector for the Hadamard transform."""
    return torch.where(
        torch.rand(shape, device=device) >= 0.5,
        torch.ones(shape, dtype=dtype, device=device),
        -torch.ones(shape, dtype=dtype, device=device),
    )


def get_hadamard_matrix(
    hadamard_dimension: int, device, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix (scaled by 1/sqrt(16))."""
    if hadamard_dimension != 16:
        raise ValueError("Only hadamard dimension 16 is supported.")
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=dtype,
            device=device,
        )
        * hadamard_scale
    )


@functools.lru_cache(maxsize=None)
def get_rht_matrix(
    sign_vector: tuple[int, ...] | None,
    device,
    dtype: torch.dtype = torch.bfloat16,
    hadamard_dimension: int = 16,
) -> torch.Tensor:
    """Construct an RHT matrix from an explicit sign vector or a generated sign vector."""
    if sign_vector is None:
        signs = get_wgrad_sign_vector(hadamard_dimension, device=device, dtype=dtype)
    else:
        if len(sign_vector) != hadamard_dimension:
            raise ValueError(
                f"Expected sign_vector length {hadamard_dimension}, "
                f"got {len(sign_vector)}"
            )
        signs = torch.tensor(sign_vector, dtype=dtype, device=device)
    sign_matrix = signs * torch.eye(hadamard_dimension, dtype=dtype, device=device)
    return sign_matrix @ get_hadamard_matrix(
        hadamard_dimension, device=device, dtype=dtype
    )


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N: tl.constexpr):
        r"""Convert flat tile_id to (pid_n, pid_m) with L2-cache-friendly grouping."""
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % num_pid_in_group) // group_size_n
        return pid_n, pid_m

    @triton.jit
    def convert_8xfp32_to_4xfp4_packed(x_pairs):
        """Convert 8 FP32 values to 4 packed FP4 bytes using round-to-nearest.
        Calls four cvt.rn instructions, each packing two FP32 values into one packed int8."""
        x_fp4x2 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints=("=r,r,r,r,r,r,r,r,r"),
            args=x_pairs,
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        return x_fp4x2

    @triton.jit
    def convert_8xfp32_to_4xfp4_packed_rs(x_pairs, rbits):
        """Convert 8 FP32 values to 4 packed FP4 bytes using stochastic rounding.
        Calls two cvt.rs instructions, each packing four FP32 values into one packed int8."""
        x_fp4x2 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 half0, half1;
            cvt.rs.satfinite.e2m1x4.f32 half0, {$6, $2, $5, $1}, $9;
            cvt.rs.satfinite.e2m1x4.f32 half1, {$8, $4, $7, $3}, $10;
            mov.b32 $0, {half0, half1};
            }
            """,
            constraints=("=r,r,r,r,r,r,r,r,r,r,r,r,r"),
            args=[x_pairs[0], x_pairs[1], rbits],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        return x_fp4x2

    @triton.jit
    def _pack_fp4(
        scaled,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        STOCHASTIC_ROUNDING: tl.constexpr,
        seed_ptr,
        offset_base_ptr,
        tile_id,
    ):
        """Pack scaled float32 values to FP4 with optional stochastic rounding.

        seed_ptr: pointer to int64 per-call seed (unused when STOCHASTIC_ROUNDING=False).
        offset_base_ptr: pointer to int64 offset base; occupies low 32 bits of the Philox
            counter (unused when STOCHASTIC_ROUNDING=False).
        tile_id: persistent-grid tile index; used to form the globally unique element index
            that occupies the high 32 bits of the Philox counter.
        """
        scaled_pairs = scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        if STOCHASTIC_ROUNDING:
            seed = tl.load(seed_ptr)
            offset_base = tl.load(offset_base_ptr)
            BLOCK_M_PACKED: tl.constexpr = BLOCK_M // 2
            local_n = tl.arange(0, BLOCK_N)[:, None]
            local_m = tl.arange(0, BLOCK_M_PACKED)[None, :]
            local_pos = local_n * BLOCK_M_PACKED + local_m
            linear_idx = tl.cast(tile_id, tl.int64) * (
                BLOCK_N * BLOCK_M_PACKED
            ) + tl.cast(local_pos, tl.int64)
            offset = (linear_idx << 32) | offset_base
            rbits = tl.randint(seed, offset)
            return convert_8xfp32_to_4xfp4_packed_rs(scaled_pairs, rbits)
        else:
            return convert_8xfp32_to_4xfp4_packed(scaled_pairs)

    @triton.jit
    def _nvfp4_quantize(
        a_t_rht, global_amax, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        """Compute per-vector FP8 scale factors and scaled FP32 values ready for FP4 packing."""
        FP8_E4M3_EPS: tl.constexpr = torch.finfo(torch.float8_e4m3fn).tiny
        FP8_E4M3_MAX: tl.constexpr = 448.0
        FP4_E2M1_MAX: tl.constexpr = 6.0
        FP32_MAX: tl.constexpr = torch.finfo(torch.float32).max

        a_vecs = tl.reshape(a_t_rht, [BLOCK_N, BLOCK_M // 16, 16])
        vec_max = tl.max(tl.abs(a_vecs), axis=-1, keep_dims=True)

        is_global_amax = global_amax == 0
        safe_global_amax = tl.where(is_global_amax, 1.0, global_amax)
        candidate = tl.minimum(FP8_E4M3_MAX * FP4_E2M1_MAX / safe_global_amax, FP32_MAX)
        candidate = tl.where(candidate == 0, 1.0, candidate)
        global_encode_scale = tl.where(is_global_amax, 1.0, candidate)
        global_decode_scale = 1.0 / global_encode_scale

        pvscale = (vec_max / FP4_E2M1_MAX) * global_encode_scale
        pvscale = tl.clamp(pvscale, FP8_E4M3_EPS, FP8_E4M3_MAX)
        pvscale_fp8 = pvscale.to(tl.float8e4nv)
        scale_inv = tl.reshape(pvscale_fp8, [BLOCK_N, BLOCK_M // 16])

        encode_scale = tl.minimum(
            1.0 / (pvscale_fp8.to(tl.float32) * global_decode_scale), FP32_MAX
        )

        scaled = a_vecs * encode_scale
        scaled = tl.clamp(scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        scaled = tl.reshape(scaled, [BLOCK_N, BLOCK_M])
        return scale_inv, scaled

    @triton.jit
    def _swizzle_scales(
        scale_inv, BLOCK_OUTER: tl.constexpr, BLOCK_INNER: tl.constexpr
    ):
        """Reshape (BLOCK_OUTER, BLOCK_INNER//16) → (BLOCK_OUTER//128, BLOCK_INNER//64, 32, 16).

        Columnwise: _swizzle_scales(scale_inv, BLOCK_N, BLOCK_M)
        Rowwise:    _swizzle_scales(scale_inv, BLOCK_M, BLOCK_N)
        """
        scale_inv = tl.reshape(
            scale_inv, [BLOCK_OUTER // 128, 4, 32, BLOCK_INNER // 64, 4]
        )
        scale_inv = tl.permute(scale_inv, [0, 3, 2, 1, 4])
        return tl.reshape(scale_inv, [BLOCK_OUTER // 128, BLOCK_INNER // 64, 32, 16])

    @triton.jit
    def _store_scales_swizzle(
        scale_inv,
        sf_ptr,
        pid_outer,
        pid_inner,
        OUTER,
        INNER,
        BLOCK_OUTER: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        """Store pre-swizzled scale factors in tile-major layout (OUTER//128, INNER//64, 32, 16).

        Columnwise: _store_scales_swizzle(sf, ptr, pid_n, pid_m, N, M, BLOCK_N, BLOCK_M)
        Rowwise:    _store_scales_swizzle(sf, ptr, pid_m, pid_n, M, N, BLOCK_M, BLOCK_N)
        """
        VEC_ELEMS_FP8: tl.constexpr = 16
        BLOCK_OUTER_TILES: tl.constexpr = BLOCK_OUTER // 128
        BLOCK_INNER_TILES: tl.constexpr = BLOCK_INNER // 64
        TILE_ELEMS: tl.constexpr = 32 * 16
        FLAT_TILE: tl.constexpr = BLOCK_OUTER_TILES * BLOCK_INNER_TILES * TILE_ELEMS

        INNER_TILES = tl.cdiv(INNER, 64)
        OUTER_TILES = tl.cdiv(OUTER, 128)

        rb_base = pid_outer * BLOCK_OUTER_TILES
        cb_base = pid_inner * BLOCK_INNER_TILES

        rb_idx = rb_base + tl.arange(0, BLOCK_OUTER_TILES)
        cb_idx = cb_base + tl.arange(0, BLOCK_INNER_TILES)
        elem_idx = tl.arange(0, TILE_ELEMS)

        offsets = (
            rb_idx[:, None, None] * INNER_TILES * TILE_ELEMS
            + cb_idx[None, :, None] * TILE_ELEMS
            + elem_idx[None, None, :]
        )
        mask = (
            (rb_idx[:, None, None] < OUTER_TILES)
            & (cb_idx[None, :, None] < INNER_TILES)
            & (elem_idx[None, None, :] < TILE_ELEMS)
        )

        flat_ptrs = sf_ptr + tl.reshape(offsets, (FLAT_TILE,))
        flat_val = tl.reshape(scale_inv, (FLAT_TILE,))
        flat_msk = tl.reshape(mask, (FLAT_TILE,))

        tl.multiple_of(flat_ptrs, VEC_ELEMS_FP8)
        tl.max_contiguous(flat_ptrs, VEC_ELEMS_FP8)
        tl.store(flat_ptrs, flat_val, mask=flat_msk)

else:

    def _compute_pid(*args, **kwargs):
        raise RuntimeError("_compute_pid requires Triton")

    def convert_8xfp32_to_4xfp4_packed(*args, **kwargs):
        raise RuntimeError("convert_8xfp32_to_4xfp4_packed requires Triton")

    def convert_8xfp32_to_4xfp4_packed_rs(*args, **kwargs):
        raise RuntimeError("convert_8xfp32_to_4xfp4_packed_rs requires Triton")

    def _pack_fp4(*args, **kwargs):
        raise RuntimeError("_pack_fp4 requires Triton")

    def _nvfp4_quantize(*args, **kwargs):
        raise RuntimeError("_nvfp4_quantize requires Triton")

    def _swizzle_scales(*args, **kwargs):
        raise RuntimeError("_swizzle_scales requires Triton")

    def _store_scales_swizzle(*args, **kwargs):
        raise RuntimeError("_store_scales_swizzle requires Triton")
