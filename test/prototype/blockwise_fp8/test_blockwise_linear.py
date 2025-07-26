# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8.blockwise_linear import Float8BlockwiseLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("in_features", [4096])
@pytest.mark.parametrize("out_features", [4096, 4 * 4096])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("block_size", [128])
def test_blockwise_quant_linear_fwd_bwd(
    in_features,
    out_features,
    batch_size,
    block_size,
):
    if in_features % block_size != 0 or out_features % block_size != 0:
        pytest.skip(f"Dimensions must be divisible by block_size={block_size}")

    torch.random.manual_seed(0)
    layer_test = Float8BlockwiseLinear(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
    ).cuda()

    torch.random.manual_seed(0)
    layer_ref = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
    ).cuda()

    # Create input tensor
    x_test = torch.randn(batch_size, 256, in_features).cuda().requires_grad_(True)
    x_ref = x_test.clone().detach().requires_grad_(True)

    # Forward pass
    y_test = layer_test(x_test)
    y_ref = layer_ref(x_ref)

    # Compare outputs
    sqnr = compute_error(y_ref, y_test)
    assert not y_test.isnan().any(), "Output must not contain NaNs"
    assert sqnr >= 25.0, f"SQNR: {sqnr.item()} must be >= 25.0"

    # Backward pass
    y_test.sum().backward()
    y_ref.sum().backward()

    # Compare input grads
    sqnr = compute_error(x_ref.grad, x_test.grad)
    assert not x_test.grad.isnan().any(), "Input grad must not contain NaNs"
    assert sqnr >= 25.0, f"SQNR: {sqnr} must be >= 25.0"

    # Compare weight grads
    sqnr = compute_error(layer_ref.weight, layer_test.weight)
    assert not layer_test.weight.grad.isnan().any(), "Weight grad must not contain NaNs"
    assert sqnr >= 25.0, f"SQNR: {sqnr} must be >= 25.0"
