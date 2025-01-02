import pytest
import torch
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_tensor import LinearMMConfig
from torchao.float8.float8_utils import is_row_major
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    hp_to_fp8_col_major,
    hp_to_fp8_row_major,
    KernelAlgorithm,
)


@pytest.mark.parametrize(
    "algo",
    [KernelAlgorithm.REDUCTION, KernelAlgorithm.ATOMIC_MAX],
)
@pytest.mark.parametrize(
    "input_shape",
    [(2, 4), (32, 16), (512, 512)],
)
def test_fp8_hp_to_fp8_row_major(input_shape: tuple[int, int], algo: KernelAlgorithm):
    assert torch.cuda.is_available()
    device = "cuda"
    input_bf16 = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    x_bf16 = input_bf16.clone().detach().to(device)
    y_bf16 = input_bf16.clone().detach().to(device)

    # production implementation
    x_fp8_row_major = hp_tensor_to_float8_dynamic(
        x_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
    )

    # float8nocompile triton implementation
    y_fp8_row_major = hp_to_fp8_row_major(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.allclose(
        x_fp8_row_major._scale, y_fp8_row_major._scale, atol=1e-3, rtol=1e-3
    )

    # check data
    assert allclose_fp8(
        x_fp8_row_major._data, y_fp8_row_major._data, atol=1e-3, rtol=1e-3
    )

    # check shapes
    assert x_fp8_row_major.shape == y_fp8_row_major.shape

    # check strides
    assert x_fp8_row_major.stride() == y_fp8_row_major.stride()

    # check memory layout
    assert is_row_major(x_fp8_row_major.stride())
    assert is_row_major(y_fp8_row_major.stride())

    # check underlying memory layout
    assert (
        x_fp8_row_major._data.storage().tolist()
        == y_fp8_row_major._data.storage().tolist()
    )

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        hp_to_fp8_row_major(
            y_bf16.t(),  # transpose so tensor memory layout is no longer contiguous
            torch.float8_e4m3fn,
            LinearMMConfig(),
        )


@pytest.mark.parametrize(
    "algo",
    [KernelAlgorithm.REDUCTION, KernelAlgorithm.ATOMIC_MAX],
)
@pytest.mark.parametrize(
    "input_shape",
    [(2, 4), (32, 16), (512, 512)],
)
def test_fp8_hp_to_fp8_col_major(input_shape: tuple[int, int], algo: KernelAlgorithm):
    assert torch.cuda.is_available()
    device = "cuda"
    input_bf16 = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    x_bf16 = input_bf16.clone().detach().to(device)
    y_bf16 = input_bf16.clone().detach().to(device)

    # production implementation
    x_fp8_row_major = hp_tensor_to_float8_dynamic(
        x_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
    )
    x_fp8_col_major = x_fp8_row_major.t().contiguous().t()

    # float8nocompile triton implementation
    y_fp8_col_major = hp_to_fp8_col_major(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.allclose(
        x_fp8_col_major._scale, y_fp8_col_major._scale, atol=1e-3, rtol=1e-3
    )

    # check data
    assert allclose_fp8(
        x_fp8_col_major._data, y_fp8_col_major._data, atol=1e-3, rtol=1e-3
    )

    # check shapes
    assert x_fp8_col_major.shape == y_fp8_col_major.shape

    # check strides
    assert x_fp8_col_major.stride() == y_fp8_col_major.stride()

    # check memory layout
    assert not is_row_major(x_fp8_col_major.stride())
    assert not is_row_major(y_fp8_col_major.stride())

    # check underlying memory layout
    assert (
        x_fp8_col_major._data.storage().tolist()
        == y_fp8_col_major._data.storage().tolist()
    )

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        hp_to_fp8_col_major(
            y_bf16.t(),  # transpose so tensor memory layout is no longer contiguous
            torch.float8_e4m3fn,
            LinearMMConfig(),
        )


def allclose_fp8(tensor1, tensor2, atol=1e-3, rtol=1e-3):
    # convert fp8 tensors to a higher precision (e.g., float32) for comparison
    # since torch.allclose does not support fp8 tensors
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for comparison.")
    if tensor1.dtype != tensor2.dtype:
        raise ValueError("Tensors must have the same dtype for comparison.")

    tensor1_fp32 = tensor1.to(torch.float32)
    tensor2_fp32 = tensor2.to(torch.float32)
    return torch.allclose(tensor1_fp32, tensor2_fp32, atol=atol, rtol=rtol)
