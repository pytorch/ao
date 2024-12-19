import torch
from torch import Tensor

from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

lib = torch.library.Library("torchao", "FRAGMENT")
lib.define(
    "quant_llm_linear(int EXPONENT, int MANTISSA, Tensor _in_feats, Tensor _weights, Tensor _scales, int splitK) -> Tensor"
)
lib.define(
    "unpack_tensor_core_tiled_layout(Tensor packed_w, int inner_k_tiles) -> Tensor"
)
lib.define(
    "dequantize_tensor_core_tiled_layout(Tensor packed_w, Tensor scales_and_zeros, int group_size, int inner_k_tiles) -> Tensor"
)
lib.define(
    "marlin_24_gemm(Tensor x, Tensor weight_marlin, Tensor meta, Tensor s, Tensor workspace, int bits, int size_m, int size_n, int size_k) -> Tensor"
)
lib.define(
    "marlin_qqq_gemm(Tensor x, Tensor weight_marlin, Tensor s_tok, Tensor s_ch, Tensor s_group, Tensor workspace, int size_m, int size_n, int size_k) -> Tensor"
)
lib.define(
    "s8s4_linear_cutlass(Tensor input, Tensor input_scale, Tensor weight, Tensor weight_scale, Tensor bias) -> Tensor"
)


def register_custom_op(name):
    def decorator(func):
        if TORCH_VERSION_AT_LEAST_2_4:
            return torch.library.register_fake(f"{name}")(func)
        else:
            return torch.library.impl_abstract(f"{name}")(func)

    return decorator


def quant_llm_linear(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = 1,
) -> Tensor:
    """
    Quant-LLM linear layer A @ W.T. See https://arxiv.org/abs/2401.14112 for more details.

    Arguments
        EXPONENT: number of exponent bits
        MANTISSA: number of mantissa bits
        _in_feats: input activations in FP16
        _weights: packed Floatx weights
        _scales: scale
        splitK: split K

    Returns
        output of linear layer
    """
    return torch.ops.torchao.quant_llm_linear.default(
        EXPONENT, MANTISSA, _in_feats, _weights, _scales, splitK
    )


@register_custom_op("torchao::quant_llm_linear")
def _(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = 1,
) -> Tensor:
    torch._check(
        _in_feats.dim() == 2,
        lambda: f"input should be a 2d tensor, got {_in_feats.dim()}D",
    )
    torch._check(
        _in_feats.dtype in (torch.float16, torch.bfloat16),
        lambda: f"weight must be FP16 or BF16, got {_in_feats.dtype}",
    )
    torch._check(
        _weights.dim() == 2,
        lambda: f"weight should be a 2d tensor, got {_weights.dim()}D",
    )
    torch._check(
        _weights.dtype is torch.uint8,
        lambda: f"weight must be UINT8, got {_weights.dtype}",
    )
    torch._check(
        _scales.dim() == 1, lambda: f"scale should be a 2d tensor, got {_scales.dim()}D"
    )
    torch._check(
        _scales.dtype in (torch.float16, torch.bfloat16),
        lambda: f"scale must be FP16 or BF16, got {_scales.dtype}",
    )

    BS, IC = _in_feats.shape
    OC, _ = _weights.shape
    N_BITS = 1 + EXPONENT + MANTISSA
    torch._check(IC // 8 * N_BITS == _weights.shape[1], lambda: "Dimensions mismatched")
    torch._check(OC == _scales.shape[0], lambda: "Dimensions mismatched")

    return _in_feats.new_empty((BS, OC))


def unpack_tensor_core_tiled_layout(packed_w: Tensor, inner_k_tiles: int) -> Tensor:
    """
    Unpacks weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`.

    Assumes that the packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `inner_k_tiles = 2 | 4 | 8`"

    Args:
        packed_w: torch.tensor: 4D tensor with shape (N / 8) x (K / (inner_k_tiles * 16)) x 32 x inner_k_tiles, dtype is torch.int32
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.int32

    """
    return torch.ops.torchao.unpack_tensor_core_tiled_layout.default(
        packed_w=packed_w, inner_k_tiles=inner_k_tiles
    )


@register_custom_op("torchao::unpack_tensor_core_tiled_layout")
def _(packed_w: Tensor, inner_k_tiles: int) -> Tensor:
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 42d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        lambda: "inner_k_tiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == inner_k_tiles / 2,
        lambda: "packed weight must have inner_k_tiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * inner_k_tiles * 16

    return torch.empty((N, K), dtype=torch.int32, device=packed_w.device)


def dequantize_tensor_core_tiled_layout(
    packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int
) -> Tensor:
    """
    Dequantizes by:
    - Unpacking weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`
    - Upcasting to bfloat16
    - Dequantizing with the scales_and_zeros that were packed with `torchao.quantization.utils.pack_tinygemm_scales_and_zeros`

    Assumes:
    - packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `inner_k_tiles = 2 | 4 | 8`"
    - packed scales_and_zeros were generated with `torchao.quantization.utils.pack_tinygemm_scales_and_zeros`
    - qGroupSize is 32 | 64 | 128 | 256

    Args:
        packed_w: torch.tensor: 4D tensor with shape `(N / 8) x (K / (inner_k_tiles * 16)) x 32 x inner_k_tiles / 2`, dtype is torch.int32
        scales_and_zeros: torch.tensor: 3D tensor with shape `numQGroups x N x 2`, dtype is torch.bfloat16 where numQGroups is K / qGroupSize
        group_size: int
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.bfloat16

    """
    return torch.ops.torchao.dequantize_tensor_core_tiled_layout.default(
        packed_w, scales_and_zeros, group_size, inner_k_tiles
    )


@register_custom_op("torchao::dequantize_tensor_core_tiled_layout")
def _(
    packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int
) -> Tensor:
    # packed_w preconditions
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 4d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        lambda: "inner_k_tiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == inner_k_tiles / 2,
        lambda: "packed weight must have inner_k_tiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * inner_k_tiles * 16

    # scales_and_zeros preconditions
    torch._check(
        scales_and_zeros.dtype is torch.bfloat16,
        lambda: "scales_and_zeros must be bfloat16",
    )
    torch._check(
        scales_and_zeros.dim() == 3,
        lambda: "scales_and_zeros must be 3D, got {scales_and_zeros.dim()}",
    )
    torch._check(
        group_size == 32 or group_size == 64 or group_size == 128 or group_size == 256,
        lambda: "qGroupSize must be 32, 64, 128, or 256",
    )
    torch._check(
        scales_and_zeros.size(0) == K // group_size,
        lambda: "scales_and_zeros must have K // qGroupSize at dim 0",
    )
    torch._check(
        scales_and_zeros.size(1) == N, lambda: "scales_and_zeros must have N at dim 1"
    )
    torch._check(
        scales_and_zeros.size(2) == 2, lambda: "scales_and_zeros must have 2 at dim 2"
    )

    return torch.empty((N, K), dtype=torch.bfloat16, device=packed_w.device)


def marlin_24_gemm(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    """
    Sparse Marlin 2:4 matrix multiplication. Reference: https://github.com/IST-DASLab/Sparse-Marlin/tree/main
    Args:
        x: input matrix of shape `(n, k/2)` in column-major layout.
        weight_marlin: weight matrix of original shape `(m, k)` in Marlin format; see `Layer.pack()`.
        meta: metadata information for 2:4 sparsity.
        s: scales of shape `(n / groupsize / 2, m)`.
        workspace: tensor with at least `m / 128 * max_par` entries that are all zero.
        bits: number of bits for quantization.
        size_m: number of rows in input matrix.
        size_n: number of columns in weight matrix.
        size_k: number of columns in input matrix.
    Returns:
        output matrix of shape `(n, m)` in column-major layout.
    """
    return torch.ops.torchao.marlin_24_gemm.default(
        x, weight_marlin, meta, s, workspace, bits, size_m, size_n, size_k
    )


@register_custom_op("torchao::marlin_24_gemm")
def _(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    TILE_SIZE = 16
    MIN_THREAD_N = 128
    MAX_PARALLELISM = 64

    # Verify num_bits
    torch._check(
        bits == 4 or bits == 8, lambda: f"num_bits must be 4 or 8. Got = {bits}"
    )
    pack_factor = 32 // bits

    # Verify M
    torch._check(
        size_m == x.size(0),
        lambda: f"Shape mismatch: x.size(0) = {x.size(0)}, size_m = {size_m}",
    )

    # Verify K
    torch._check(
        size_k == x.size(1),
        lambda: f"Shape mismatch: x.size(1) = {x.size(1)}, size_k = {size_k}",
    )
    torch._check(
        size_k % TILE_SIZE == 0,
        lambda: f"size_k = {size_k} is not divisible by tile_size = {TILE_SIZE}",
    )
    torch._check(
        (size_k // TILE_SIZE // 2) == weight_marlin.size(0),
        lambda: f"Shape mismatch: weight_marlin.size(0) = {weight_marlin.size(0)}, size_k = {size_k}, tile_size = {TILE_SIZE}",
    )

    # Verify N
    torch._check(
        s.size(1) == size_n, lambda: f"s.size(1) = {s.size(1)}, size_n = {size_n}"
    )
    torch._check(
        weight_marlin.size(1) % TILE_SIZE == 0,
        lambda: f"weight_marlin.size(1) = {weight_marlin.size(1)} is not divisible by tile_size = {TILE_SIZE}",
    )

    actual_size_n = (weight_marlin.size(1) // TILE_SIZE) * pack_factor
    torch._check(
        size_n == actual_size_n,
        lambda: f"size_n = {size_n}, actual_size_n = {actual_size_n}",
    )

    # Verify meta
    torch._check(
        meta.size(0) == size_k // 8 // 2 // 2,
        lambda: f"meta.size(0) = {meta.size(0)} is not size_k / 8 / 2 / 2 = {size_k // 8 // 2 // 2}",
    )
    torch._check(
        meta.size(1) == size_n * 2,
        lambda: f"meta.size(1) = {meta.size(1)} is not size_n * 2 = {size_n * 2}",
    )

    # Verify A device and strides
    torch._check(x.is_cuda, lambda: "x is not on GPU")
    torch._check(x.is_contiguous(), lambda: "x is not contiguous")

    # Verify B device and strides
    torch._check(weight_marlin.is_cuda, lambda: "weight_marlin is not on GPU")
    torch._check(
        weight_marlin.is_contiguous(), lambda: "weight_marlin is not contiguous"
    )

    # Verify meta device and strides
    torch._check(meta.is_cuda, lambda: "meta is not on GPU")
    torch._check(meta.is_contiguous(), lambda: "meta is not contiguous")

    # Verify scales device and strides
    torch._check(s.is_cuda, lambda: "s is not on GPU")
    torch._check(s.is_contiguous(), lambda: "s is not contiguous")

    # Verify groupsize
    groupsize = -1
    if s.size(0) > 1:
        torch._check(
            size_k % s.size(0) == 0,
            lambda: f"size_k = {size_k} is not divisible by s.size(0) = {s.size(0)}",
        )
        groupsize = size_k // s.size(0)
        groupsize //= 2  # Because of 24
    torch._check(
        groupsize == -1 or groupsize == 64,
        lambda: f"Unexpected groupsize = {groupsize}",
    )

    # Verify workspace size
    torch._check(
        size_n % MIN_THREAD_N == 0,
        lambda: f"size_n = {size_n} is not divisible by min_thread_n = {MIN_THREAD_N}",
    )
    min_workspace_size = (size_n // MIN_THREAD_N) * MAX_PARALLELISM
    torch._check(
        workspace.numel() >= min_workspace_size,
        lambda: f"workspace.numel = {workspace.numel()} is below min_workspace_size = {min_workspace_size}",
    )

    return torch.empty((x.size(0), s.size(1)), dtype=x.dtype, device=x.device)


def marlin_qqq_gemm(
    x: Tensor,
    weight_marlin: Tensor,
    s_tok: Tensor,
    s_ch: Tensor,
    s_group: Tensor,
    workspace: Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    """
    Marlin for W4A8 mixed precision matrix multiplication.
    See https://arxiv.org/pdf/2406.09904 for more details.
    Reference: https://github.com/HandH1998/QQQ/tree/main
    Args:
        x: `torch.int8` input matrix of shape `(m, k)` in standard row-major layout.
        weight_marlin: `torch.int32` weight matrix of original shape `(k, n)` in the specified format.
        s_tok: `torch.float32` activation per-token quantization scales of shape `(m, 1)`.
        s_ch: `torch.float32` weight per-channel quantization scales of shape `(1, n)`.
        s_group: `torch.half` weight per-group quantization scales of shape `(m / groupsize, n)`, it should be empty when group_size != -1.
        workspace: `torch.int32` tensor with at least `n / 128 * max_par` entries that are all zero.
        size_m: number of rows in input matrix.
        size_n: number of columns in weight matrix.
        size_k: number of columns in input matrix.
    Returns:
        `torch.half` out matrix of shape `(m, n)` in standard row-major layout.
    """
    return torch.ops.torchao.marlin_qqq_gemm.default(
        x, weight_marlin, s_tok, s_ch, s_group, workspace, size_m, size_n, size_k
    )


@register_custom_op("torchao::marlin_qqq_gemm")
def _(
    x: Tensor,
    weight_marlin: Tensor,
    s_tok: Tensor,
    s_ch: Tensor,
    s_group: Tensor,
    workspace: Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    TILE_SIZE = 16
    MIN_THREAD_N = 64
    MAX_PARALLELISM = 16
    PACK_FACTOR = 32 // 4

    # Verify M
    torch._check(
        size_m == x.size(0),
        lambda: f"Shape mismatch: x.size(0) = {x.size(0)}, size_m = {size_m}",
    )
    torch._check(
        size_m == s_tok.numel(),
        lambda: f"Shape mismatch: s_tok.numel() = {s_tok.numel()}, size_m = {size_m}",
    )

    # Verify K
    torch._check(
        size_k == x.size(1),
        lambda: f"Shape mismatch: x.size(1) = {x.size(1)}, size_k = {size_k}",
    )
    torch._check(
        size_k % TILE_SIZE == 0,
        lambda: f"size_k = {size_k} is not divisible by tile_size = {TILE_SIZE}",
    )
    torch._check(
        (size_k // TILE_SIZE) == weight_marlin.size(0),
        lambda: f"Shape mismatch: weight_marlin.size(0) = {weight_marlin.size(0)}, size_k = {size_k}, tile_size = {TILE_SIZE}",
    )

    # Verify groupsize
    groupsize = -1 if s_group.numel() == 0 else size_k // s_group.size(0)
    torch._check(groupsize in [-1, 128], lambda: f"Unexpected groupsize = {groupsize}")

    # Verify N
    torch._check(
        s_ch.numel() == size_n,
        lambda: f"Shape mismatch: s_ch.numel() = {s_ch.numel()}, size_n = {size_n}",
    )
    torch._check(
        weight_marlin.size(1) % TILE_SIZE == 0,
        lambda: f"weight_marlin.size(1) = {weight_marlin.size(1)} is not divisible by tile_size = {TILE_SIZE}",
    )
    if groupsize != -1:
        torch._check(
            s_group.size(1) == size_n,
            lambda: f"Shape mismatch: s_group.size(1) = {s_group.size(1)}, size_n = {size_n}",
        )
        torch._check(
            size_k % s_group.size(0) == 0,
            lambda: f"size_k = {size_k} is not divisible by s_group.size(0) = {s_group.size(0)}",
        )

    actual_size_n = (weight_marlin.size(1) // TILE_SIZE) * PACK_FACTOR
    torch._check(
        size_n == actual_size_n,
        lambda: f"Shape mismatch: size_n = {size_n}, actual_size_n = {actual_size_n}",
    )

    # Verify A device and strides
    torch._check(x.is_cuda, lambda: "x is not on GPU")
    torch._check(x.is_contiguous(), lambda: "x is not contiguous")

    # Verify B device and strides
    torch._check(weight_marlin.is_cuda, lambda: "weight_marlin is not on GPU")
    torch._check(
        weight_marlin.is_contiguous(), lambda: "weight_marlin is not contiguous"
    )

    # Verify s_tok device, strides and dtype
    torch._check(s_tok.is_cuda, lambda: "s_tok is not on GPU")
    torch._check(s_tok.is_contiguous(), lambda: "s_tok is not contiguous")
    torch._check(s_tok.dtype == torch.float32, lambda: "s_tok's dtype is not float32")

    # Verify s_ch device, strides and dtype
    torch._check(s_ch.is_cuda, lambda: "s_ch is not on GPU")
    torch._check(s_ch.is_contiguous(), lambda: "s_ch is not contiguous")
    torch._check(s_ch.dtype == torch.float32, lambda: "s_ch's dtype is not float32")

    # Verify s_group device, strides and dtype
    torch._check(s_group.is_cuda, lambda: "s_group is not on GPU")
    torch._check(s_group.is_contiguous(), lambda: "s_group is not contiguous")
    torch._check(s_group.dtype == torch.float16, "s_group's dtype is not float16")

    # Verify workspace size
    torch._check(
        size_n % MIN_THREAD_N == 0,
        lambda: f"size_n = {size_n} is not divisible by min_thread_n = {MIN_THREAD_N}",
    )
    min_workspace_size = (size_n // MIN_THREAD_N) * MAX_PARALLELISM
    torch._check(
        workspace.numel() >= min_workspace_size,
        lambda: f"workspace.numel() = {workspace.numel()} is below min_workspace_size = {min_workspace_size}",
    )

    return torch.empty((size_m, size_n), dtype=torch.float16, device=x.device)


def s8s4_linear_cutlass(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor,
) -> Tensor:
    """
    CUTLASS-based W4A8 linear operator.
    Args:
        input: input tensor, quantized to 8-bit integer values.
        input_scale: scale factors for input tensor, has to be tensor of the same shape as the input tensor, minus the last dimension.
        weight: weight matrix, quantized to 4-bit integer values, in row-major layout.
        weight_scale: scale factors for weight tensor, one value per row of weight matrix (thus also tensor of the same shape as the weight tensor, minus the last dimension).
        bias: a vector of size equal to number of rows of weight tensor, or None.
    Returns:
        output: result tensor, in row-major layout.
    """

    return torch.ops.torchao.s8s4_linear_cutlass.default(
        input, input_scale, weight, weight_scale, bias
    )


@register_custom_op("torchao::s8s4_linear_cutlass")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor,
) -> Tensor:
    # Validate dtypes.
    torch._check(
        input.dtype == torch.int8,
        lambda: f"input dtype {input.dtype} instead of {torch.int8}",
    )
    torch._check(
        input_scale.dtype in (torch.float16, torch.bfloat16),
        lambda: f"input_scale dtype {input_scale.dtype} instead of {torch.float16} or {torch.bfloat16}",
    )
    torch._check(
        weight.dtype == torch.int8,
        lambda: f"weight dtype {weight.dtype} instead of {torch.int8}",
    )
    torch._check(
        weight_scale.dtype == input_scale.dtype,
        lambda: f"weight_scale dtype {weight_scale.dtype} instead of {input_scale.dtype}",
    )
    if bias is not None:
        torch._check(
            bias.dtype == input_scale.dtype,
            lambda: f"bias dtype {weight_scale.dtype} instead of {input_scale.dtype}",
        )

    # Validate dims.
    torch._check(input.dim() >= 2, lambda: f"input is {input.dim()}D instead of >=2D")
    torch._check(
        input_scale.dim() == input.dim() - 1,
        lambda: f"input_scale is {input_scale.dim()}D instead of {input.dim() - 1}D",
    )
    torch._check(weight.dim() == 2, lambda: f"weight is {weight.dim()}D instead of 2D")
    torch._check(
        weight_scale.dim() == 1 or weight_scale.dim() == 2,
        lambda: f"weight_scale is {weight_scale.dim()}D instead of 1D or 2D",
    )
    if bias is not None:
        torch._check(bias.dim() == 1, lambda: f"bias is {bias.dim()}D instead of 1D")

    # Validate shapes.
    torch._check(
        input.shape[-1] == 2 * weight.shape[-1],
        lambda: "input and weight shapes do not match for matrix product",
    )
    for i in range(input_scale.dim()):
        torch._check(
            input_scale.shape[i] == input.shape[i],
            lambda: f"input_scale and input shapes do not match at position {i}",
        )
    torch._check(
        weight_scale.numel() == weight.shape[0],
        lambda: f"weight_scale has {weight_scale.numel()} elements instead of {weight.shape[0]}",
    )
    if bias is not None:
        torch._check(
            bias.numel() == weight.shape[0],
            lambda: f"bias has {bias.numel()} elements instead of {weight.shape[0]}",
        )

    # Validate strides (input, input_scales and weight_scales will be
    # reshape()-d by the operator, so no need to check strides for
    # them).
    torch._check(weight.stride(-1) == 1, lambda: "weight is not in row-major layout")
    if bias is not None:
        torch._check(bias.is_contiguous(), lambda: "bias is not contiguous")

    return torch.empty(
        (*input.shape[:-1], weight.shape[0]),
        dtype=input_scale.dtype,
        device=input.device,
    )
