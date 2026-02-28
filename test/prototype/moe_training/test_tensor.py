# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from torchao.utils import torch_version_at_least

# Skip module if basic requirements aren't met
if not (torch_version_at_least("2.7.0") and torch.cuda.is_available()):
    pytest.skip("CUDA and PyTorch 2.7.0+ required", allow_module_level=True)

from torchao.prototype.moe_training.config import (
    MXFP8TrainingConfig,
    MXFP8TrainingRecipe,
)
from torchao.prototype.moe_training.tensor import TorchAOTrainingTensor
from torchao.quantization.utils import compute_error


@pytest.mark.parametrize("op_name", ["mm", "matmul", "linear"])
@pytest.mark.parametrize("batch_size", [None, 2, 4])
def test_mxfp8_training_tensor_ops_fwd_bwd(op_name, batch_size):
    # mm doesn't support batching
    if op_name == "mm" and batch_size is not None:
        pytest.skip("mm doesn't support batching")

    config = MXFP8TrainingConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL)

    # Create input tensors - dimensions must be divisible by 32
    # Use larger sizes for better SQNR, especially with bias in linear ops
    M, K, N = 1024, 1024, 2048
    if batch_size is None:
        A_shape = (M, K)
    else:
        A_shape = (batch_size, M, K)

    A = torch.randn(*A_shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    bias = (
        torch.randn(N, dtype=torch.bfloat16, device="cuda")
        if op_name == "linear"
        else None
    )

    # Reference computation with bf16
    A_ref = A.clone().detach().requires_grad_(True)
    B_ref = B.clone().detach().requires_grad_(True)

    if op_name == "mm":
        result_ref = torch.mm(A_ref, B_ref.t())
    elif op_name == "matmul":
        result_ref = torch.matmul(A_ref, B_ref.t())
    elif op_name == "linear":
        result_ref = F.linear(A_ref, B_ref, bias)

    # MXFP8 computation
    B_mxfp8 = TorchAOTrainingTensor(B, config)

    if op_name == "mm":
        result_mxfp8 = torch.mm(A, B_mxfp8)
    elif op_name == "matmul":
        result_mxfp8 = torch.matmul(A, B_mxfp8)
    elif op_name == "linear":
        result_mxfp8 = F.linear(A, B_mxfp8, bias)

    # Validate forward pass
    assert result_mxfp8.shape == result_ref.shape, "Shape mismatch"
    assert result_mxfp8.dtype == torch.bfloat16, "Dtype should be bfloat16"
    assert not isinstance(result_mxfp8, TorchAOTrainingTensor), (
        "Result should be unwrapped"
    )

    # Check forward SQNR
    # Linear with bias has slightly lower SQNR due to bias addition
    sqnr_fwd = compute_error(result_ref, result_mxfp8)
    min_sqnr_fwd = 26.0 if op_name == "linear" else 27.0
    assert sqnr_fwd >= min_sqnr_fwd, (
        f"Forward SQNR {sqnr_fwd} is too low, must be >= {min_sqnr_fwd}"
    )

    # Backward pass with MSE loss to avoid contiguity issues
    labels_ref = torch.ones_like(result_ref)
    labels_mxfp8 = torch.ones_like(result_mxfp8)
    loss_ref = F.mse_loss(result_ref, labels_ref)
    loss_mxfp8 = F.mse_loss(result_mxfp8, labels_mxfp8)
    loss_ref.backward()
    loss_mxfp8.backward()

    # Verify gradients exist
    assert A.grad is not None, "A.grad should be computed"
    assert A_ref.grad is not None, "A_ref.grad should be computed"
    assert B_mxfp8.grad is not None, "B_mxfp8.grad should be computed"
    assert B_ref.grad is not None, "B_ref.grad should be computed"

    # Check input gradient SQNR
    sqnr_input_grad = compute_error(A_ref.grad, A.grad)
    min_sqnr_input_grad = 25.0
    assert sqnr_input_grad >= min_sqnr_input_grad, (
        f"Input grad SQNR {sqnr_input_grad} is too low, must be >= {min_sqnr_input_grad}"
    )

    # Check weight gradient SQNR
    sqnr_weight_grad = compute_error(B_ref.grad, B_mxfp8.grad)
    min_sqnr_weight_grad = 24.0
    assert sqnr_weight_grad >= min_sqnr_weight_grad, (
        f"Weight grad SQNR {sqnr_weight_grad} is too low, must be >= {min_sqnr_weight_grad}"
    )


def test_mxfp8_training_tensor_ops_preserve_subclass():
    config = MXFP8TrainingConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL)

    B = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    B_mxfp8 = TorchAOTrainingTensor(B, config)

    # view
    result = B_mxfp8.view(32, 64)
    assert isinstance(result, TorchAOTrainingTensor), "view should preserve subclass"

    # transpose.int
    result = B_mxfp8.transpose(0, 1)
    assert isinstance(result, TorchAOTrainingTensor), (
        "transpose.int should preserve subclass"
    )

    # transpose.default
    result = B_mxfp8.t()
    assert isinstance(result, TorchAOTrainingTensor), (
        "transpose.default should preserve subclass"
    )

    # clone
    result = B_mxfp8.clone()
    assert isinstance(result, TorchAOTrainingTensor), "clone should preserve subclass"

    # slice
    result = B_mxfp8[:32, :]
    assert isinstance(result, TorchAOTrainingTensor), "slice should preserve subclass"
