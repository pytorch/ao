# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# This actually belongs to test_ops.py, extracted here for easier
# maintenance.

import itertools

import pytest
import torch
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.optests import opcheck

from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
    _float8_cutlass_quant_sparse,
)
from torchao.sparsity.utils import create_semi_structured_tensor
from torchao.testing.utils import skip_if_rocm

DTYPES = [torch.float16, torch.bfloat16]
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
TEST_PARAMS = list(
    itertools.product(
        DTYPES,
        XQ_WQ_DTYPES,
        BATCH_SIZE,
        SIZE_MNK,
        USE_BIAS,
    )
)


def run_test_for_op(
    op,
    dtype,
    Xq_dtype,
    Wq_dtype,
    batch_size,
    size_mnk,
    use_bias,
):
    device = "cuda"

    size_m, size_n, size_k = size_mnk

    X = torch.randn((batch_size, size_m, size_k), dtype=dtype, device=device)
    W = create_semi_structured_tensor(size_n, size_k, dtype=dtype).to(device)
    bias = torch.rand((size_n,), dtype=dtype, device=device) if use_bias else None

    X_quant_func = _float8_cutlass_quant
    W_quant_func = _float8_cutlass_quant_sparse
    X_aqt = X_quant_func(X, Xq_dtype)
    W_aqt = W_quant_func(W, Wq_dtype)

    Xq = X_aqt.tensor_impl.float8_data
    X_scale = X_aqt.tensor_impl.scale
    Wq_sparse = W_aqt.tensor_impl.sparse
    W_meta = W_aqt.tensor_impl.meta
    W_scale = W_aqt.tensor_impl.scale
    Wq_dense, _, _ = W_aqt.tensor_impl.get_plain()

    # If torch.nn.functional.linear(X, W, bias) used as reference, the
    # error would be too big.  The calculation below is approximately
    # what rowwise_scaled_linear_sparse_cutlass kernel is doing.
    output_ref = (Xq.float() @ Wq_dense.float().T) * X_scale[..., None] * W_scale
    if bias is not None:
        output_ref += bias
    output_ref = output_ref.to(dtype).reshape(X.shape[:-1] + (size_n,))

    fn_inputs = (Xq, X_scale, Wq_sparse, W_meta, W_scale, bias, dtype)
    try:
        output = op(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("operator not implemented")

    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=5e-3)

    # Perform opcheck.
    test_utils = [
        "test_autograd_registration",
        "test_faketensor",
    ]  # "test_schema" not implemented for FP8 data types
    opcheck(
        op,
        fn_inputs,
        test_utils=test_utils,
    )


@skip_if_rocm("does not yet work on ROCm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not SM90OrLater, reason="FP8 is only supported on H100+ devices")
@pytest.mark.parametrize(
    "dtype, Xq_Wq_dtypes, batch_size, size_mnk, use_bias",
    TEST_PARAMS,
)
def test_rowwise_scaled_linear_sparse_cutlass_f8f8(
    dtype,
    Xq_Wq_dtypes,
    batch_size,
    size_mnk,
    use_bias,
):
    run_test_for_op(
        torch.ops.torchao.rowwise_scaled_linear_sparse_cutlass_f8f8,
        dtype,
        *Xq_Wq_dtypes,
        batch_size,
        size_mnk,
        use_bias,
    )
