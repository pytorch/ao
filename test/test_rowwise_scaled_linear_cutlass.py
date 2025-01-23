import itertools

import pytest
import torch

from torchao.ops import (
    rowwise_scaled_linear_cutlass_s4s4,
    rowwise_scaled_linear_cutlass_s8s4,
)
from torchao.quantization.utils import group_quantize_tensor_symmetric

ROWWISE_SCALED_LINEAR_CUTLASS_DTYPE = [torch.float16, torch.bfloat16]
ROWWISE_SCALED_LINEAR_CUTLASS_BATCH_SIZE = [1, 4, 8, 16, 32, 64]
ROWWISE_SCALED_LINEAR_CUTLASS_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]
ROWWISE_SCALED_LINEAR_CUTLASS_USE_BIAS = [False, True]
ROWWISE_SCALED_LINEAR_CUTLASS_TEST_PARAMS = list(
    itertools.product(
        ROWWISE_SCALED_LINEAR_CUTLASS_DTYPE,
        ROWWISE_SCALED_LINEAR_CUTLASS_BATCH_SIZE,
        ROWWISE_SCALED_LINEAR_CUTLASS_SIZE_MNK,
        ROWWISE_SCALED_LINEAR_CUTLASS_USE_BIAS,
    )
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype, batch_size, size_mnk, use_bias", ROWWISE_SCALED_LINEAR_CUTLASS_TEST_PARAMS
)
def test_rowwise_scaled_linear_cutlass_s4s4(dtype, batch_size, size_mnk, use_bias):
    size_m, size_n, size_k = size_mnk

    input = torch.randn((batch_size, size_m, size_k), dtype=dtype, device="cuda")
    weight = torch.rand((size_n, size_k), dtype=dtype, device="cuda")
    bias = torch.rand((size_n,), dtype=dtype, device="cuda") if use_bias else None

    input_2d = input.view(-1, input.shape[-1])
    input_2d_s8, input_2d_scales, input_2d_zeros = group_quantize_tensor_symmetric(
        input_2d, 4, size_k, dtype
    )
    assert torch.all(input_2d_zeros == 0)
    input_s8 = input_2d_s8.reshape(input.shape)
    input_s4 = (input_s8[..., 1::2] << 4) | (input_s8[..., 0::2] & 0xF)
    input_scales = input_2d_scales.reshape(input.shape[:-1])

    weight_s8, weight_scales, weight_zeros = group_quantize_tensor_symmetric(
        weight, 4, size_n, dtype
    )
    assert torch.all(weight_zeros == 0)
    weight_s4 = (weight_s8[:, 1::2] << 4) | (weight_s8[:, 0::2] & 0xF)

    # If torch.nn.functional.linear(input, weight, bias) used as
    # reference, the error would be too big.  The calculation below is
    # approximately what rowwise_scaled_linear_cutlass kernel is doing
    # (except that matrix multiplication is over integers there)).
    size_m_2d = input_2d.shape[0]
    output_ref = (
        (input_2d_s8.float() @ weight_s8.float().T)
        * input_2d_scales.view(size_m_2d, 1)
        * weight_scales.view(1, size_n)
    )
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.to(dtype).reshape(input.shape[:-1] + (size_n,))

    fn_inputs = (input_s4, input_scales, weight_s4, weight_scales, bias)
    try:
        output = rowwise_scaled_linear_cutlass_s4s4(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("rowwise_scaled_linear_cutlass() op not implemented")

    torch.testing.assert_close(output, output_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype, batch_size, size_mnk, use_bias", ROWWISE_SCALED_LINEAR_CUTLASS_TEST_PARAMS
)
def test_rowwise_scaled_linear_cutlass_s8s4(dtype, batch_size, size_mnk, use_bias):
    size_m, size_n, size_k = size_mnk

    input = torch.randn((batch_size, size_m, size_k), dtype=dtype, device="cuda")
    weight = torch.rand((size_n, size_k), dtype=dtype, device="cuda")
    bias = torch.rand((size_n,), dtype=dtype, device="cuda") if use_bias else None

    input_2d = input.view(-1, input.shape[-1])
    input_2d_s8, input_2d_scales, input_2d_zeros = group_quantize_tensor_symmetric(
        input_2d, 8, size_k, dtype
    )
    assert torch.all(input_2d_zeros == 0)
    input_s8 = input_2d_s8.reshape(input.shape)
    input_scales = input_2d_scales.reshape(input.shape[:-1])

    weight_s8, weight_scales, weight_zeros = group_quantize_tensor_symmetric(
        weight, 4, size_n, dtype
    )
    assert torch.all(weight_zeros == 0)
    weight_s4 = ((weight_s8[:, 1::2] & 0xF) << 4) | (weight_s8[:, 0::2] & 0xF)

    # If torch.nn.functional.linear(input, weight, bias) used as
    # reference, the error would be too big.  The calculation below is
    # approximately what rowwise_scaled_linear_cutlass kernel is doing
    # (except that matrix multiplication is over integers there)).
    size_m_2d = input_2d.shape[0]
    output_ref = (
        (input_2d_s8.float() @ weight_s8.float().T)
        * input_2d_scales.view(size_m_2d, 1)
        * weight_scales.view(1, size_n)
    )
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.to(dtype).reshape(input.shape[:-1] + (size_n,))

    fn_inputs = (input_s8, input_scales, weight_s4, weight_scales, bias)
    try:
        output = rowwise_scaled_linear_cutlass_s8s4(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("rowwise_scaled_linear_cutlass() op not implemented")

    torch.testing.assert_close(output, output_ref)
