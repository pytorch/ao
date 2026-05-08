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
    Float8TrainingOpConfig,
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
)
from torchao.prototype.moe_training.tensor import (
    Float8TrainingWeightWrapperTensor,
    MXFP8TrainingWeightWrapperTensor,
)
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
)
from torchao.prototype.mx_formats.kernels import (
    _mxfp8_cuda_kernels_available,
    _triton_kernels_available,
)
from torchao.quantization.utils import compute_error
from torchao.utils import is_sm_at_least_100


@pytest.mark.parametrize("op_name", ["mm", "matmul", "linear"])
@pytest.mark.parametrize("batch_size", [None, 2, 4])
@pytest.mark.parametrize(
    "recipe",
    [MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL, MXFP8TrainingRecipe.MXFP8_RCEIL],
)
@pytest.mark.parametrize(
    "cast_kernel_choice",
    [MXFP8Dim1CastKernelChoice.CUDA, MXFP8Dim1CastKernelChoice.CUTEDSL],
)
def test_mxfp8_training_tensor_ops_fwd_bwd(
    op_name, batch_size, recipe, cast_kernel_choice
):
    if recipe != MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL:
        if not is_sm_at_least_100() or not _triton_kernels_available:
            pytest.skip("SM 100+ required for real MXFP8 support")
        if (
            cast_kernel_choice == MXFP8Dim1CastKernelChoice.CUDA
            and not _mxfp8_cuda_kernels_available
        ):
            pytest.skip("MXFP8 CUDA kernels not available")
        if (
            cast_kernel_choice == MXFP8Dim1CastKernelChoice.CUTEDSL
            and not _mxfp8_cutedsl_kernels_available
        ):
            pytest.skip("MXFP8 CUTEDSL kernels not available")

    # mm doesn't support batching
    if op_name == "mm" and batch_size is not None:
        pytest.skip("mm doesn't support batching")

    config = MXFP8TrainingOpConfig.from_recipe(recipe)
    config.dim1_cast_kernel_choice = cast_kernel_choice

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
    B_mxfp8 = MXFP8TrainingWeightWrapperTensor(B, config)

    if op_name == "mm":
        result_mxfp8 = torch.mm(A, B_mxfp8)
    elif op_name == "matmul":
        result_mxfp8 = torch.matmul(A, B_mxfp8)
    elif op_name == "linear":
        result_mxfp8 = F.linear(A, B_mxfp8, bias)

    # Validate forward pass
    assert result_mxfp8.shape == result_ref.shape, "Shape mismatch"
    assert result_mxfp8.dtype == torch.bfloat16, "Dtype should be bfloat16"
    assert not isinstance(result_mxfp8, MXFP8TrainingWeightWrapperTensor), (
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
    config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL)

    B = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    B_mxfp8 = MXFP8TrainingWeightWrapperTensor(B, config)

    # view
    result = B_mxfp8.view(32, 64)
    assert isinstance(result, MXFP8TrainingWeightWrapperTensor), (
        "view should preserve subclass"
    )

    # transpose.int
    result = B_mxfp8.transpose(0, 1)
    assert isinstance(result, MXFP8TrainingWeightWrapperTensor), (
        "transpose.int should preserve subclass"
    )

    # transpose.default
    result = B_mxfp8.t()
    assert isinstance(result, MXFP8TrainingWeightWrapperTensor), (
        "transpose.default should preserve subclass"
    )

    # clone
    result = B_mxfp8.clone()
    assert isinstance(result, MXFP8TrainingWeightWrapperTensor), (
        "clone should preserve subclass"
    )

    # slice
    result = B_mxfp8[:32, :]
    assert isinstance(result, MXFP8TrainingWeightWrapperTensor), (
        "slice should preserve subclass"
    )


@pytest.mark.parametrize("op_name", ["mm", "matmul", "linear"])
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize(
    "float8_linear_recipe", ["tensorwise", "rowwise", "rowwise_with_gw_hp"]
)
def test_float8_training_tensor_ops_fwd_bwd(op_name, batch_size, float8_linear_recipe):
    # mm doesn't support batching
    if op_name == "mm" and batch_size is not None:
        pytest.skip("mm doesn't support batching")

    # All FP8 linear recipes require SM89+ (torch._scaled_mm)
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 linear requires SM89+")

    # rowwise and rowwise_with_gw_hp require SM90+ (CUTLASS axiswise kernels)
    if float8_linear_recipe in (
        "rowwise",
        "rowwise_with_gw_hp",
    ) and torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Rowwise FP8 requires SM90+")

    config = Float8TrainingOpConfig(float8_linear_recipe=float8_linear_recipe)

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

    # FP8 computation
    B_fp8 = Float8TrainingWeightWrapperTensor(B, config)

    if op_name == "mm":
        result_fp8 = torch.mm(A, B_fp8)
    elif op_name == "matmul":
        result_fp8 = torch.matmul(A, B_fp8)
    elif op_name == "linear":
        result_fp8 = F.linear(A, B_fp8, bias)

    # Validate forward pass
    assert result_fp8.shape == result_ref.shape, "Shape mismatch"
    assert result_fp8.dtype == torch.bfloat16, "Dtype should be bfloat16"
    assert not isinstance(result_fp8, Float8TrainingWeightWrapperTensor), (
        "Result should be unwrapped"
    )

    # Check forward SQNR
    sqnr_fwd = compute_error(result_ref, result_fp8)
    min_sqnr_fwd = 25.0
    assert sqnr_fwd >= min_sqnr_fwd, (
        f"Forward SQNR {sqnr_fwd} is too low, must be >= {min_sqnr_fwd}"
    )

    # Backward pass
    labels_ref = torch.ones_like(result_ref)
    labels_fp8 = torch.ones_like(result_fp8)
    loss_ref = F.mse_loss(result_ref, labels_ref)
    loss_fp8 = F.mse_loss(result_fp8, labels_fp8)
    loss_ref.backward()
    loss_fp8.backward()

    # Verify gradients exist
    assert A.grad is not None, "A.grad should be computed"
    assert A_ref.grad is not None, "A_ref.grad should be computed"
    assert B_fp8.grad is not None, "B_fp8.grad should be computed"
    assert B_ref.grad is not None, "B_ref.grad should be computed"

    # Check input gradient SQNR
    sqnr_input_grad = compute_error(A_ref.grad, A.grad)
    min_sqnr_input_grad = 24.0
    assert sqnr_input_grad >= min_sqnr_input_grad, (
        f"Input grad SQNR {sqnr_input_grad} is too low, must be >= {min_sqnr_input_grad}"
    )

    # Check weight gradient SQNR
    sqnr_weight_grad = compute_error(B_ref.grad, B_fp8.grad)
    min_sqnr_weight_grad = 23.0
    assert sqnr_weight_grad >= min_sqnr_weight_grad, (
        f"Weight grad SQNR {sqnr_weight_grad} is too low, must be >= {min_sqnr_weight_grad}"
    )
