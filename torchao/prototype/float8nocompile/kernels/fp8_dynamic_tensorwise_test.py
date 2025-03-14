# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_tensor import LinearMMConfig
from torchao.float8.float8_utils import is_row_major
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
    hp_to_fp8_col_major,
    hp_to_fp8_col_major_t,
    hp_to_fp8_col_major_t_and_non_t,
    hp_to_fp8_row_and_col_major,
    hp_to_fp8_row_major,
    hp_to_fp8_row_major_t,
    hp_to_fp8_row_major_t_and_non_t,
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
    assert torch.eq(x_fp8_row_major._scale, y_fp8_row_major._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_row_major._data, y_fp8_row_major._data))

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
def test_fp8_hp_to_fp8_row_major_t(input_shape: tuple[int, int], algo: KernelAlgorithm):
    assert torch.cuda.is_available()
    device = "cuda"
    input_bf16 = torch.tensor(
        [[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16, device=device
    )
    x_bf16 = input_bf16.clone().detach().to(device)
    y_bf16 = input_bf16.clone().detach().to(device)

    # production implementation
    x_fp8_row_major = hp_tensor_to_float8_dynamic(
        x_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
    )
    x_fp8_row_major_t = x_fp8_row_major.t().contiguous()

    # float8nocompile triton implementation
    y_fp8_row_major_t = hp_to_fp8_row_major_t(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.eq(x_fp8_row_major_t._scale, y_fp8_row_major_t._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_row_major_t._data, y_fp8_row_major_t._data))

    # check shapes
    assert x_fp8_row_major_t.shape == y_fp8_row_major_t.shape

    # check strides
    assert x_fp8_row_major_t.stride() == y_fp8_row_major_t.stride()

    # check memory layout
    assert is_row_major(x_fp8_row_major_t.stride())
    assert is_row_major(y_fp8_row_major_t.stride())

    # check underlying memory layout
    assert (
        x_fp8_row_major_t._data.storage().tolist()
        == y_fp8_row_major_t._data.storage().tolist()
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
    assert torch.eq(x_fp8_col_major._scale, y_fp8_col_major._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_col_major._data, y_fp8_col_major._data))

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


@pytest.mark.parametrize(
    "algo",
    [KernelAlgorithm.REDUCTION, KernelAlgorithm.ATOMIC_MAX],
)
@pytest.mark.parametrize(
    "input_shape",
    [(2, 4), (32, 16), (512, 512)],
)
def test_fp8_hp_to_fp8_col_major_t(input_shape: tuple[int, int], algo: KernelAlgorithm):
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
    x_fp8_col_major_t = x_fp8_row_major.contiguous().t()

    # float8nocompile triton implementation
    y_fp8_col_major_t = hp_to_fp8_col_major_t(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.eq(x_fp8_col_major_t._scale, y_fp8_col_major_t._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_col_major_t._data, y_fp8_col_major_t._data))

    # check shapes
    assert x_fp8_col_major_t.shape == y_fp8_col_major_t.shape

    # check strides
    assert x_fp8_col_major_t.stride() == y_fp8_col_major_t.stride()

    # check memory layout
    assert not is_row_major(x_fp8_col_major_t.stride())
    assert not is_row_major(y_fp8_col_major_t.stride())

    # check underlying memory layout
    assert (
        x_fp8_col_major_t._data.storage().tolist()
        == y_fp8_col_major_t._data.storage().tolist()
    )

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        hp_to_fp8_col_major(
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
def test_fp8_hp_to_fp8_row_and_col_major(
    input_shape: tuple[int, int], algo: KernelAlgorithm
):
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
    y_fp8_row_major, y_fp8_col_major = hp_to_fp8_row_and_col_major(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.eq(x_fp8_row_major._scale, y_fp8_row_major._scale)
    assert torch.eq(x_fp8_col_major._scale, y_fp8_col_major._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_row_major._data, y_fp8_row_major._data))
    assert torch.all(torch.eq(x_fp8_col_major._data, y_fp8_col_major._data))

    # check shapes
    assert x_fp8_row_major.shape == y_fp8_row_major.shape
    assert x_fp8_col_major.shape == y_fp8_col_major.shape

    # check strides
    assert x_fp8_row_major.stride() == y_fp8_row_major.stride()
    assert x_fp8_col_major.stride() == y_fp8_col_major.stride()

    # check memory layout
    assert is_row_major(x_fp8_row_major.stride())
    assert is_row_major(y_fp8_row_major.stride())
    assert not is_row_major(x_fp8_col_major.stride())
    assert not is_row_major(y_fp8_col_major.stride())

    # check underlying memory layout
    assert (
        x_fp8_row_major._data.storage().tolist()
        == y_fp8_row_major._data.storage().tolist()
    )
    assert (
        x_fp8_col_major._data.storage().tolist()
        == y_fp8_col_major._data.storage().tolist()
    )

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        hp_to_fp8_row_and_col_major(
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
def test_fp8_hp_to_fp8_row_major_t_and_non_t(
    input_shape: tuple[int, int], algo: KernelAlgorithm
):
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
    x_fp8_row_major_t = x_fp8_row_major.t().contiguous()

    # float8nocompile triton implementation
    y_fp8_row_major, y_fp8_row_major_t = hp_to_fp8_row_major_t_and_non_t(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.eq(x_fp8_row_major._scale, y_fp8_row_major._scale)
    assert torch.eq(x_fp8_row_major_t._scale, y_fp8_row_major_t._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_row_major._data, y_fp8_row_major._data))
    assert torch.all(torch.eq(x_fp8_row_major_t._data, y_fp8_row_major_t._data))

    # check shapes
    assert x_fp8_row_major.shape == y_fp8_row_major.shape
    assert x_fp8_row_major_t.shape == y_fp8_row_major_t.shape

    # check strides
    assert x_fp8_row_major.stride() == y_fp8_row_major.stride()
    assert x_fp8_row_major_t.stride() == y_fp8_row_major_t.stride()

    # check memory layout
    assert is_row_major(x_fp8_row_major.stride())
    assert is_row_major(y_fp8_row_major.stride())
    assert is_row_major(x_fp8_row_major_t.stride())
    assert is_row_major(y_fp8_row_major_t.stride())

    # check underlying memory layout
    assert (
        x_fp8_row_major._data.storage().tolist()
        == y_fp8_row_major._data.storage().tolist()
    )
    assert (
        x_fp8_row_major_t._data.storage().tolist()
        == y_fp8_row_major_t._data.storage().tolist()
    )

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        hp_to_fp8_row_major_t_and_non_t(
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
def test_fp8_hp_to_fp8_col_major_t_and_non_t(
    input_shape: tuple[int, int], algo: KernelAlgorithm
):
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
    x_fp8_col_major_t = x_fp8_row_major.t()

    # float8nocompile triton implementation
    y_fp8_col_major, y_fp8_col_major_t = hp_to_fp8_col_major_t_and_non_t(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    # check scales
    assert torch.eq(x_fp8_col_major._scale, y_fp8_col_major._scale)
    assert torch.eq(x_fp8_col_major_t._scale, y_fp8_col_major_t._scale)

    # check data
    assert torch.all(torch.eq(x_fp8_col_major._data, y_fp8_col_major._data))
    assert torch.all(torch.eq(x_fp8_col_major_t._data, y_fp8_col_major_t._data))

    # check shapes
    assert x_fp8_col_major.shape == y_fp8_col_major.shape
    assert x_fp8_col_major_t.shape == y_fp8_col_major_t.shape

    # check strides
    assert x_fp8_col_major.stride() == y_fp8_col_major.stride()
    assert x_fp8_col_major_t.stride() == y_fp8_col_major_t.stride()

    # check memory layout
    assert not is_row_major(x_fp8_col_major.stride())
    assert not is_row_major(y_fp8_col_major.stride())
    assert not is_row_major(x_fp8_col_major_t.stride())
    assert not is_row_major(y_fp8_col_major_t.stride())

    # check underlying memory layout
    assert (
        x_fp8_col_major._data.storage().tolist()
        == y_fp8_col_major._data.storage().tolist()
    )
    assert (
        x_fp8_col_major_t._data.storage().tolist()
        == y_fp8_col_major_t._data.storage().tolist()
    )

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        hp_to_fp8_col_major_t_and_non_t(
            y_bf16.t(),  # transpose so tensor memory layout is no longer contiguous
            torch.float8_e4m3fn,
            LinearMMConfig(),
        )
