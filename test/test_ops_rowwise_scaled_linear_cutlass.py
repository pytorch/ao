# This actually belongs to test_ops.py, extracted here for easier
# maintenance.

import itertools

import pytest
import torch
from torch.testing._internal.optests import opcheck

from torchao.quantization.quant_api import (
    _int4_symm_cutlass_quant,
    _int8_symm_cutlass_quant,
)

DTYPES = [torch.float16, torch.bfloat16]
BATCH_SIZE = [1, 4, 8, 16, 32, 64]
SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]
USE_BIAS = [False, True]
TEST_PARAMS = list(
    itertools.product(
        DTYPES,
        BATCH_SIZE,
        SIZE_MNK,
        USE_BIAS,
    )
)


def run_test_for_op(op, dtype, batch_size, size_mnk, use_bias):
    size_m, size_n, size_k = size_mnk

    X = torch.randn((batch_size, size_m, size_k), dtype=dtype, device="cuda")
    W = torch.rand((size_n, size_k), dtype=dtype, device="cuda")
    bias = torch.rand((size_n,), dtype=dtype, device="cuda") if use_bias else None

    Xq_bits = 4 if op == torch.ops.torchao.rowwise_scaled_linear_cutlass_s4s4 else 8

    X_quant_func = (
        _int4_symm_cutlass_quant if Xq_bits == 4 else _int8_symm_cutlass_quant
    )
    W_quant_func = _int4_symm_cutlass_quant
    X_aqt = X_quant_func(X)
    W_aqt = W_quant_func(W)

    Xq = X_aqt.tensor_impl.int_data
    X_scale = X_aqt.tensor_impl.scale
    Wq = W_aqt.tensor_impl.int_data
    W_scale = W_aqt.tensor_impl.scale
    Xq_int8, _, _ = X_aqt.tensor_impl.get_plain()
    Wq_int8, _, _ = W_aqt.tensor_impl.get_plain()

    # If torch.nn.functional.linear(X, W, bias) used as reference, the
    # error would be too big.  The calculation below is approximately
    # what rowwise_scaled_linear_cutlass kernel is doing.
    output_ref = (Xq_int8.float() @ Wq_int8.float().T) * X_scale[..., None] * W_scale
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.to(dtype).reshape(X.shape[:-1] + (size_n,))

    fn_inputs = (Xq, X_scale, Wq, W_scale, bias, dtype)
    try:
        output = op(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("operator not implemented")

    torch.testing.assert_close(output, output_ref)

    # Perform opcheck.
    test_utils = ["test_schema", "test_autograd_registration", "test_faketensor"]
    opcheck(
        op,
        fn_inputs,
        test_utils=test_utils,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype, batch_size, size_mnk, use_bias", TEST_PARAMS)
def test_rowwise_scaled_linear_cutlass_s4s4(dtype, batch_size, size_mnk, use_bias):
    run_test_for_op(
        torch.ops.torchao.rowwise_scaled_linear_cutlass_s4s4,
        dtype,
        batch_size,
        size_mnk,
        use_bias,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype, batch_size, size_mnk, use_bias", TEST_PARAMS)
def test_rowwise_scaled_linear_cutlass_s8s4(dtype, batch_size, size_mnk, use_bias):
    run_test_for_op(
        torch.ops.torchao.rowwise_scaled_linear_cutlass_s8s4,
        dtype,
        batch_size,
        size_mnk,
        use_bias,
    )
