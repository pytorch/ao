import itertools

import pytest
import torch

from torchao.ops import s8s4_linear_cutlass
from torchao.quantization.utils import group_quantize_tensor_symmetric
from torchao.utils import compute_max_diff

S8S4_LINEAR_CUTLASS_DTYPE = [torch.float16, torch.bfloat16]
S8S4_LINEAR_CUTLASS_BATCH_SIZE = [1, 4, 8, 16, 32, 64]
S8S4_LINEAR_CUTLASS_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]
S8S4_LINEAR_CUTLASS_USE_BIAS = [False, True]
S8S4_LINEAR_CUTLASS_TEST_PARAMS = list(
    itertools.product(
        S8S4_LINEAR_CUTLASS_DTYPE,
        S8S4_LINEAR_CUTLASS_BATCH_SIZE,
        S8S4_LINEAR_CUTLASS_SIZE_MNK,
        S8S4_LINEAR_CUTLASS_USE_BIAS,
    )
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype, batch_size, size_mnk, use_bias", S8S4_LINEAR_CUTLASS_TEST_PARAMS
)
def test_s8s4_linear_cutlass(dtype, batch_size, size_mnk, use_bias):
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
    # approximately what s8s4_linear_cutlass kernel is doing (except
    # that matrrix multiplication is over integers there)).
    size_m_2d = input_2d.shape[0]
    output_ref = (
        (input_2d_s8.to(dtype) @ weight_s8.to(dtype).T)
        * input_2d_scales.view(size_m_2d, 1)
        * weight_scales.view(1, size_n)
    )
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.reshape(input.shape[:-1] + (size_n,))

    fn_inputs = (input_s8, input_scales, weight_s4, weight_scales, bias)
    try:
        output = s8s4_linear_cutlass(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("s8s4_linear_cutlass() op not implemented")

    max_diff = compute_max_diff(output, output_ref)
    assert max_diff < 5e-3
