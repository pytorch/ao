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


def run_test_for_op(op, xq_bits, wq_bits, dtype, batch_size, size_mnk, use_bias):
    assert xq_bits in [4, 8]
    assert wq_bits in [4, 8]

    size_m, size_n, size_k = size_mnk

    x = torch.randn((batch_size, size_m, size_k), dtype=dtype, device="cuda")
    w = torch.rand((size_n, size_k), dtype=dtype, device="cuda")
    bias = torch.rand((size_n,), dtype=dtype, device="cuda") if use_bias else None

    x_2d = x.view(-1, x.shape[-1])
    xq_2d_s8, xq_2d_scales, xq_2d_zeros = group_quantize_tensor_symmetric(
        x_2d, xq_bits, size_k, dtype
    )
    assert torch.all(xq_2d_zeros == 0)
    xq_s8 = xq_2d_s8.reshape(x.shape)
    if xq_bits == 4:
        xq = (xq_s8[..., 1::2] << 4) | (xq_s8[..., 0::2] & 0xF)
    else:
        xq = xq_s8
    xq_scales = xq_2d_scales.reshape(x.shape[:-1])

    wq_s8, wq_scales, wq_zeros = group_quantize_tensor_symmetric(
        w, wq_bits, size_n, dtype
    )
    assert torch.all(wq_zeros == 0)
    if wq_bits == 4:
        wq = (wq_s8[:, 1::2] << 4) | (wq_s8[:, 0::2] & 0xF)
    else:
        wq = wq_s8

    # If torch.nn.functional.linear(x, w, bias) used as reference, the
    # error would be too big.  The calculation below is approximately
    # what rowwise_scaled_linear_cutlass kernel is doing (except that
    # matrix multiplication is over integers there).
    size_m_2d = x_2d.shape[0]
    output_ref = (
        (xq_2d_s8.float() @ wq_s8.float().T)
        * xq_2d_scales.view(size_m_2d, 1)
        * wq_scales.view(1, size_n)
    )
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.to(dtype).reshape(x.shape[:-1] + (size_n,))

    fn_inputs = (xq, xq_scales, wq, wq_scales, bias)
    try:
        output = op(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("operator not implemented")

    torch.testing.assert_close(output, output_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype, batch_size, size_mnk, use_bias", ROWWISE_SCALED_LINEAR_CUTLASS_TEST_PARAMS
)
def test_rowwise_scaled_linear_cutlass_s4s4(dtype, batch_size, size_mnk, use_bias):
    run_test_for_op(
        rowwise_scaled_linear_cutlass_s4s4, 4, 4, dtype, batch_size, size_mnk, use_bias
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype, batch_size, size_mnk, use_bias", ROWWISE_SCALED_LINEAR_CUTLASS_TEST_PARAMS
)
def test_rowwise_scaled_linear_cutlass_s8s4(dtype, batch_size, size_mnk, use_bias):
    run_test_for_op(
        rowwise_scaled_linear_cutlass_s8s4, 8, 4, dtype, batch_size, size_mnk, use_bias
    )
