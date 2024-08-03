# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn.utils.parametrize as parametrize
from torchao.utils import find_multiple
from .quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    int_scaled_matmul,
)
from torchao.utils import TORCH_VERSION_AFTER_2_5

__all__ = [
    "compute_error",
    "_apply_logging_hook",
    "quantize_activation_per_token_absmax",
    "quant_int8_dynamic_per_token_linear",
    "quant_int8_per_token_matmul",
    "dynamically_quantize_per_channel",
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "get_groupwise_affine_qparams",
    "pack_tinygemm_scales_and_zeros",
    "unpack_tinygemm_scales_and_zeros",
    "groupwise_affine_quantize_tensor_from_qparams",
    "groupwise_affine_dequantize_tensor_from_qparams",
    "groupwise_affine_quantize_tensor",
    "groupwise_affine_dequantize_tensor",
    "per_token_dynamic_quant",
    "get_group_qparams_symmetric",
    "recommended_inductor_config_setter"
]

try:
    import lm_eval  # pyre-ignore[21]  # noqa: F401

    _lm_eval_available = True
except:
    _lm_eval_available = False

# basic SQNR
def compute_error(x, y):
    Ps = torch.linalg.norm(x)
    Pn = torch.linalg.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# logger for fqn + op + shape
# note: not safe for any kind of multithreading
_cur_fqn: Optional[str] = None


def _get_logging_hook(fqn):

    def forward_hook(module, input):
        global _cur_fqn
        _cur_fqn = fqn

    return forward_hook


def _apply_logging_hook(model):
    for name, mod in model.named_modules():
        mod.register_forward_pre_hook(_get_logging_hook(name))


# collections.defaultdict printing is weird with lambdas, so hand writing for now
_fqn_to_op_to_shape_to_count: Dict[
    Optional[str], Dict[Optional[str], Dict[Optional[str], int]]
] = {}


class LoggingTensorMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        global _cur_fqn
        op_name: str = f"{func.__module__}.{func.__name__}"
        shape_str = ""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape_str += str(list(arg.shape)) + ", "
        if shape_str != "":
            shape_str = shape_str[:-2]

        if _cur_fqn not in _fqn_to_op_to_shape_to_count:
            _fqn_to_op_to_shape_to_count[_cur_fqn] = {}
        if op_name not in _fqn_to_op_to_shape_to_count[_cur_fqn]:
            _fqn_to_op_to_shape_to_count[_cur_fqn][op_name] = {}
        if shape_str not in _fqn_to_op_to_shape_to_count[_cur_fqn][op_name]:
            _fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] = 0
        _fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] += 1

        return rs

class _MultiInput:

    def __init__(self, inputs):

        self.values = list(inputs)

    def add_input(self, input):
        self.values.append(input)
        return self

    def __getitem__(self, slice):
        return _MultiInput(self.values[slice])

    def cuda(self):
        self.values = [
            val.cuda() if isinstance(val, torch.Tensor) else val for val in self.values
        ]


def guard_dtype_size(tensor_arg, arg_name, dtype=None, size=None):
    if dtype is not None and tensor_arg.dtype != dtype:
        raise ValueError(f"Expected Tensor argument {arg_name} to have dtype {dtype}, but got {tensor_arg.dtype} instead.")
    if size is not None and tensor_arg.size() != size:
        raise ValueError(f"Expected Tensor argument {arg_name} to have size {size}, but got {tensor_arg.size()} instead.")

def _get_per_token_block_size(x: torch.Tensor) -> List[int]:
    block_size = []
    for _ in range(len(x.shape)-1):
        block_size.append(1)
    block_size.append(x.shape[-1])
    return block_size

# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified
def quantize_activation_per_token_absmax(t):
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
    mapping_type = MappingType.SYMMETRIC
    block_size = list(t.shape)
    for i in range(len(block_size) - 1):
        block_size[i] = 1
    dtype = torch.int8
    eps = 1e-5
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    quant_min = -127
    quant_max = 127
    scale_dtype = torch.float32 if t.dtype == torch.float16 else None

    scale, zero_point = choose_qparams_affine(t, mapping_type, block_size, dtype, quant_min, quant_max, eps, scale_dtype=scale_dtype)

    quantized = quantize_affine(t, block_size, scale, zero_point, dtype, quant_min, quant_max)

    return quantized, scale

def quant_int8_dynamic_per_token_linear(
    x,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype,
):
    """
    like F.linear, but with int8 dynamic quantization of activation,
    and a quantized weight
    """
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = quant_int8_per_token_matmul(
        x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype
    )
    if bias is not None:
        mm_out = mm_out + bias
    return mm_out

def quant_int8_per_token_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8_t,
    w_scales,
    output_dtype=torch.float32,
):
    """
    Quantized matmul of int8 operands that accumulates to int32 and returns
    output_dtype. For now, this is written for approximate numerical
    Assumes that activation and weight quantization are symmetric,
    i.e. act_zp and w_zp is 0.
    Assumes that weight quantization is per-channel.

    see
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    for an overview of quantized matmul compute

    in scalar form, assuming output_dtype is fp32 and zw == 0:

      Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    """

    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #
    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_scaled = int_scaled_matmul(tmp, w_vals_int8_t, x_scales.reshape(-1, 1))

    y = (y_dot_scaled * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
    )

    # can downcast only at the very end
    y = y.to(output_dtype)
    return y

def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    """
    assumes symmetric quantization
    assumes axis == 0
    assumes dense memory format
    TODO(future): relax ^ as needed
    """

    assert x.dim() == 2, "only support 2d Tensors"

    eps = torch.finfo(torch.float32).eps
    block_size = (1, x.shape[1])
    zero_point_dtype = torch.int64

    mapping_type = MappingType.SYMMETRIC
    scale, zero_point = choose_qparams_affine(x, mapping_type, block_size, target_dtype=target_dtype, quant_min=quant_min, quant_max=quant_max, eps=eps, zero_point_dtype=zero_point_dtype)
    quant = quantize_affine(x, block_size, scale, zero_point, target_dtype, quant_min, quant_max)
    return quant, scale, zero_point

# reference: https://fburl.com/code/vfsygwd0
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    block_size = int_repr.shape
    input_dtype = int_repr.dtype
    assert scale.numel() == 1, f"scale size: {scale.numel()}"
    dequantized = dequantize_affine(int_repr, block_size, scale, zero_point, input_dtype, output_dtype=out_dtype)
    return dequantized


# reference: https://fburl.com/code/org0fmi3
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    assert int_repr.dim() == 2, "only support 2d Tensors"
    # channel axis == 0
    # block_size before transpose should be (1, int_repr.shape[1]) for axis == 0 per channel quant

    # TODO: transpose is for perf reasons for torch.compile, we should separate this to lowering step
    int_repr = int_repr.t()
    # transpose for block_size as well
    block_size = (int_repr.shape[0], 1)
    input_dtype = int_repr.dtype
    dequantized = dequantize_affine(int_repr, block_size, scales, zero_points, input_dtype, output_dtype=out_dtype)
    dequantized = dequantized.t()
    return dequantized

def get_groupwise_affine_qparams(w, n_bit=4, groupsize=128, dtype=torch.bfloat16):
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2
    assert n_bit <= 8, f"only n_bit smaller than 8 is supported, got: {n_bit}"

    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.int32
    block_size = (1, groupsize)
    quant_min = 0
    quant_max = 2**n_bit - 1
    eps = 1e-6
    scale_dtype = dtype
    zero_point_dtype = dtype

    scale, zero_point = choose_qparams_affine(
        w,
        mapping_type,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=False,
        zero_point_domain=ZeroPointDomain.FLOAT
    )

    return scale.to(dtype=dtype).reshape(w.shape[0], -1), zero_point.to(
        dtype=dtype
    ).reshape(w.shape[0], -1)


def pack_tinygemm_scales_and_zeros(scales, zeros, dtype=torch.bfloat16):
    guard_dtype_size(scales, "scales", dtype=dtype, size=zeros.size())
    guard_dtype_size(zeros, "zeros", dtype=dtype)
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_tinygemm_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def groupwise_affine_quantize_tensor_from_qparams(
    w,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    block_size = (1, groupsize)
    output_dtype = torch.int32
    quant_min = 0
    quant_max = 2 ** n_bit - 1

    int_data = quantize_affine(w, block_size, scales, zeros, output_dtype, quant_min, quant_max, zero_point_domain = ZeroPointDomain.FLOAT)
    if TORCH_VERSION_AFTER_2_5:
        int_data_device_type = int_data.device.type
        # Move to cpu, until issue with MPS memory management of temporary tensors is resolved
        if int_data_device_type == 'mps':
            int_data = int_data.cpu()
        int_data = (int_data[::, ::2] << 4 | int_data[::, 1::2]).to(torch.uint8)
        if int_data_device_type == 'mps':
            int_data = int_data.to(device='mps')
    return int_data

def groupwise_affine_dequantize_tensor_from_qparams(
    w_int4x8,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    assert w_int4x8.dim() == 2
    if TORCH_VERSION_AFTER_2_5:
        data = w_int4x8.to(torch.int32)
        high_bits = data >> 4
        low_bits = data & 0x0F
        w_int32 = torch.zeros((w_int4x8.shape[0], w_int4x8.shape[1] * 2), dtype=torch.int32, device=w_int4x8.device)
        w_int32[::, ::2] = high_bits
        w_int32[::, 1::2] = low_bits
    else:
        w_int32 = w_int4x8

    # needed for GPTQ single column dequantize
    if groupsize > w_int32.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int32.shape[-1]
    assert w_int32.shape[-1] % groupsize == 0
    block_size = (1, groupsize)
    input_dtype = torch.int32
    quant_min = 0
    quant_max = 2**n_bit - 1
    return dequantize_affine(w_int32, block_size, scales, zeros, input_dtype, quant_min, quant_max, zero_point_domain=ZeroPointDomain.FLOAT, output_dtype=scales.dtype)

def groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128, dtype=torch.bfloat16):
    scales, zeros = get_groupwise_affine_qparams(w, n_bit, groupsize, dtype)
    w_int4x8 = groupwise_affine_quantize_tensor_from_qparams(
        w, scales, zeros, n_bit, groupsize
    )
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros, dtype)
    return w_int4x8, scales_and_zeros


def groupwise_affine_dequantize_tensor(
    w_int4x8,
    scales_and_zeros,
    n_bit=4,
    groupsize=128,
):
    scales, zeros = unpack_tinygemm_scales_and_zeros(scales_and_zeros)
    return groupwise_affine_dequantize_tensor_from_qparams(
        w_int4x8, scales, zeros, n_bit, groupsize
    )


# TODO: separate scale and zero point precision
def get_group_qparams_symmetric(w, n_bit=4, groupsize=128, precision=torch.float32):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2
    assert n_bit <= 8, f"unsupported n_bit: {n_bit}"

    mapping_type = MappingType.SYMMETRIC
    block_size = (1, groupsize)
    eps = torch.finfo(torch.float32).eps
    ranges = {}
    ranges[1] = (-1, 0)
    # generating ranges for bit 2 to 8
    for i in range(2, 9):
        ranges[i] = (-(2 ** (i - 1)), 2 ** (i - 1) - 1)
    quant_min, quant_max = ranges[n_bit]
    scale, zero_point = choose_qparams_affine(w, mapping_type, block_size, target_dtype=torch.int8, quant_min=quant_min, quant_max=quant_max, eps=eps, scale_dtype=precision, zero_point_dtype=precision)
    return scale.reshape(w.shape[0], -1), zero_point.reshape(w.shape[0], -1)


def group_quantize_tensor_symmetric(
    w,
    n_bit=4,
    group_size=128,
    precision=torch.float32,
):
    scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size, precision)
    n_bit = 4
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    # TODO: currently we don't know how to express torch.int4, we'll
    # add torch.int4 to core later
    from torchao._executorch_ops import _quantized_decomposed_quantize_per_channel_group_wrapper
    w_int8 = _quantized_decomposed_quantize_per_channel_group_wrapper(
        w, scales, zeros, min_int, max_int, torch.int8, group_size
    )

    return w_int8, scales, zeros

def per_token_dynamic_quant(input: torch.Tensor) -> torch.Tensor:
    orig_dtype = input.dtype
    # TODO: we may need to make the choose_qparams op configurable
    from torchao._executorch_ops import _quantized_decomposed_choose_qparams_per_token_asymmetric_wrapper
    (
        scales,
        zero_points,
    ) = _quantized_decomposed_choose_qparams_per_token_asymmetric_wrapper(
        input, torch.int8
    )

    # TODO: get these from torch.int8
    quant_min = -128
    quant_max = 127
    from torchao._executorch_ops import _quantized_decomposed_quantize_per_token_wrapper
    input = _quantized_decomposed_quantize_per_token_wrapper(
        input, scales, zero_points, quant_min, quant_max, torch.int8
    )
    from torchao._executorch_ops import _quantized_decomposed_dequantize_per_token_wrapper
    input = _quantized_decomposed_dequantize_per_token_wrapper(
        input, scales, zero_points, quant_min, quant_max, torch.int8, orig_dtype
    )
    return input.to(orig_dtype)

def recommended_inductor_config_setter():
    """
    Set inductor config to use the following optimizations which have been showed to improve performance for quantized models:
        coordinate_descent_tuning = True
        coordinate_descent_check_all_directions = True
        force_fuse_int_mm_with_mul = True
        fx_graph_cache = True
        triton.unique_kernel_names = True
        torch.set_float32_matmul_precision("high")
    """
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch.set_float32_matmul_precision("high")
