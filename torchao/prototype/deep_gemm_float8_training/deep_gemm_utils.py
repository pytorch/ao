# TODO gate by existence of deep_gemm library
import deep_gemm
import torch


def scaled_mm_deep_gemm_128_1_128_128(a, b, a_scale, b_scale):
    M, K = a.shape
    N, K = b.shape
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    deep_gemm.gemm_fp8_fp8_bf16_nt((a, a_scale), (b, b_scale), out=out)
    return out


def scaled_mm_deep_gemm_128_1_128_1(a, b, a_scale, b_scale):
    M, K = a.shape
    N, K = b.shape
    # Note: the results from `wgrad_gemm_fp8_fp8_fp32_nt` are **accumulated**
    # into this tensor. For now, we initialize with `zeros` to get correct
    # numerics in toy examples. For a real use case, this will need to pass
    # in the gradient tensor directly.
    out = torch.zeros((M, N), dtype=torch.float, device=a.device)
    deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt((a, a_scale), (b, b_scale), out=out)
    return out


def scale_narrow_tiles(x, tile_size=128):
    """
    Input: weight tensor in high precision
    Output: weight tensor in float8, and scale, tiled 1 by tile_size

    This is one function because logically this should be a fused kernel.
    """
    # TODO assert row major
    orig_shape = x.shape
    x = x.reshape(-1, tile_size)
    x_amax = x.abs().max(dim=1).values.unsqueeze(1).clamp(1e-4)
    # TODO read from finfo instead of hardcoding
    s = 448.0 / x_amax

    x = (x * s).clamp(min=-448.0, max=448.0).to(torch.float8_e4m3fn)
    x = x.reshape(*orig_shape)
    s = s.reshape(orig_shape[0], -1).to(torch.float)
    return x, s


def unscale_narrow_tiles(x, s, tile_size=128):
    # for debugging
    orig_shape = x.shape
    x = x.reshape(-1, tile_size)
    s = s.reshape(-1).unsqueeze(1)
    x = x.to(torch.float) / s
    x = x.reshape(*orig_shape)
    return x


def scale_square_tiles(x, tile_size=128):
    """
    Input: weight tensor in high precision
    Output: weight tensor in float8, and scale, tiled tile_size by tile_size

    This is one function because logically this should be a fused kernel.
    `torch.compile` currently has three kernels, we should write a triton
    to speed this up kernel and file an issue for compile to catch up.
    """
    # TODO assert row major
    assert len(x.shape) == 2, "unsupported"
    height, width = x.shape

    # might be funky with dynamic shapes...
    t_h = height // tile_size
    t_w = width // tile_size
    x = x.reshape(t_h, tile_size, t_w, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, tile_size * tile_size)
    m = x.abs().max(dim=1).values.unsqueeze(1).clamp(1e-4)

    # convert to scale
    # TODO read from finfo instead of hardcoding
    s = 448.0 / m

    x = (x * s).clamp(min=-448.0, max=448.0).to(torch.float8_e4m3fn)
    x = x.reshape(t_h, t_w, tile_size, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(height, width)
    s = s.reshape(t_h, t_w).to(torch.float)

    return x, s


def unscale_square_tiles(x, s, tile_size=128):
    # for debugging

    assert len(x.shape) == 2, "unsupported"
    height, width = x.shape

    # might be funky with dynamic shapes...
    t_h = height // tile_size
    t_w = width // tile_size
    x = x.reshape(t_h, tile_size, t_w, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, tile_size * tile_size)

    s = s.reshape(-1).unsqueeze(1)

    x = x.float() / s

    x = x.reshape(t_h, t_w, tile_size, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(height, width)
    return x
