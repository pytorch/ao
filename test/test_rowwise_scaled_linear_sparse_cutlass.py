import itertools

import pytest
import torch
from torch.testing._internal.common_cuda import SM90OrLater

from torchao.dtypes import (
    Float8Layout,
    to_affine_quantized_floatx,
)
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
    to_sparse_semi_structured_cutlass_sm9x_f8,
)
from torchao.quantization.utils import _get_per_token_block_size
from torchao.sparsity.utils import create_semi_structured_tensor

X_W_DTYPES = [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)]
XQ_WQ_DTYPES = [
    (torch.float8_e4m3fn, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float8_e5m2),
    (torch.float8_e5m2, torch.float8_e4m3fn),
    (torch.float8_e5m2, torch.float8_e5m2),
]
BATCH_SIZE = [1, 4]
SIZE_MNK = [
    (2, 128, 256),
    (3, 128, 256),
    (13, 128, 256),
    (27, 128, 128),
    (33, 128, 64),
    (65, 128, 32),
]
USE_BIAS = [False, True]
BIAS_DTYPE = [torch.float16]
TEST_PARAMS = list(
    itertools.product(
        X_W_DTYPES,
        XQ_WQ_DTYPES,
        BATCH_SIZE,
        SIZE_MNK,
        USE_BIAS,
        BIAS_DTYPE,
    )
)


def run_test_for_op(
    op,
    x_dtype,
    w_dtype,
    xq_dtype,
    wq_dtype,
    batch_size,
    size_mnk,
    use_bias,
    bias_dtype,
):
    size_m, size_n, size_k = size_mnk

    x = torch.randn((batch_size, size_m, size_k), dtype=x_dtype, device="cuda")
    w = create_semi_structured_tensor(size_n, size_k, dtype=w_dtype)
    bias = torch.rand((size_n,), dtype=bias_dtype, device="cuda") if use_bias else None

    x_aqt = to_affine_quantized_floatx(
        input_float=x,
        target_dtype=xq_dtype,
        block_size=_get_per_token_block_size(x),
        _layout=Float8Layout(mm_config=None),
    )
    xq, xq_scales, zero_points = x_aqt.tensor_impl.get_plain()
    assert zero_points is None

    w_aqt = to_affine_quantized_floatx(
        input_float=w,
        target_dtype=wq_dtype,
        block_size=_get_per_token_block_size(w),
        _layout=Float8Layout(mm_config=None),
    )
    wq, wq_scales, zero_points = w_aqt.tensor_impl.get_plain()
    assert zero_points is None
    wq_sp, wq_sp_meta = to_sparse_semi_structured_cutlass_sm9x_f8(wq)
    wq_sp_scales = wq_scales

    xq_2d = xq.view(-1, xq.shape[-1])
    size_m_2d = xq_2d.shape[0]
    output_ref = (
        (xq_2d.float() @ wq.float().T)
        * xq_scales.view(size_m_2d, 1)
        * wq_scales.view(1, size_n)
    )
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.to(x.dtype).reshape(x.shape[:-1] + (size_n,))

    fn_inputs = (xq, xq_scales, wq_sp, wq_sp_meta, wq_sp_scales, bias)
    try:
        output = op(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("operator not implemented")

    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not SM90OrLater, reason="FP8 is only supported on H100+ devices")
@pytest.mark.parametrize(
    "x_w_dtypes, xq_wq_dtypes, batch_size, size_mnk, use_bias, bias_dtype",
    TEST_PARAMS,
)
def test_rowwise_scaled_liner_sparse_cutlass_f8f8(
    x_w_dtypes,
    xq_wq_dtypes,
    batch_size,
    size_mnk,
    use_bias,
    bias_dtype,
):
    run_test_for_op(
        rowwise_scaled_linear_sparse_cutlass_f8f8,
        *x_w_dtypes,
        *xq_wq_dtypes,
        batch_size,
        size_mnk,
        use_bias,
        bias_dtype,
    )
