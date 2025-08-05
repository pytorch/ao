# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch

from torchao.utils import is_sm_at_least_90

triton = pytest.importorskip("triton", reason="Triton required to run this test")
if torch.cuda.is_available and not is_sm_at_least_90():
    pytest.skip("This test requires SM90 or higher", allow_module_level=True)


from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8_training.linear import Float8BlockwiseLinear

torch.random.manual_seed(0)

from torchao.utils import auto_detect_device

_DEVICE = auto_detect_device()

@pytest.mark.parametrize("in_features", [4096])
@pytest.mark.parametrize("out_features", [128256])
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

    layer_ref = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
    ).to(_DEVICE)

    layer_test = Float8BlockwiseLinear.from_float(copy.deepcopy(layer_ref))

    # Create input tensor
    x_test = torch.randn(batch_size, 256, in_features).to(_DEVICE).requires_grad_(True)
    x_ref = x_test.clone().detach().requires_grad_(True)

    # Forward pass
    y_test = layer_test(x_test)
    y_ref = layer_ref(x_ref)

    # Compare outputs
    sqnr = compute_error(y_ref, y_test)
    assert not y_test.isnan().any(), "Output must not contain NaNs"
    assert sqnr >= 25.0, f"SQNR: {sqnr.item()} must be >= 25.0"
    assert not sqnr.isinf().any(), "SQNR must not be inf"

    # Backward pass
    y_test.sum().backward()
    y_ref.sum().backward()

    # Compare input grads
    sqnr = compute_error(x_ref.grad, x_test.grad)
    assert not x_test.grad.isnan().any(), "Input grad must not contain NaNs"
    assert sqnr >= 30.0, f"SQNR: {sqnr} must be >= 25.0"

    # Compare weight grads
    sqnr = compute_error(layer_ref.weight, layer_test.weight)
    assert not layer_test.weight.grad.isnan().any(), "Weight grad must not contain NaNs"
    assert sqnr >= 30.0, f"SQNR: {sqnr} must be >= 25.0"
