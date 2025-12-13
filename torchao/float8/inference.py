# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Defines an nn module designed to be used during inference
"""

import math
from typing import List, NamedTuple, Optional, Tuple, Union

import torch

from torchao.float8.float8_utils import is_row_major, pad_tensor_for_matmul
from torchao.float8.types import FP8Granularity
from torchao.quantization.granularity import (
    PerBlock,
    PerRow,
    PerTensor,
)
from torchao.utils import (
    is_MI300,
    is_sm_at_least_89,
)

Tensor = torch.Tensor


class Float8MMConfig(NamedTuple):
    """
    Configuration for the scaled_mm in the forward and backward pass.

    Attributes:
        emulate (bool): Whether to emulate the matmuls in fp32.
        use_fast_accum (bool): Whether to use the fast-accumulation option for scaled_mm.
        pad_inner_dim (bool): Whether to pad the inner dimension of a and b with 0s.
                              This is needed for matmuls not aligned to 16.
    """

    emulate: bool = False
    use_fast_accum: bool = False
    pad_inner_dim: bool = False


def preprocess_data(
    a_data: Tensor,
    b_data: Tensor,
    scaled_mm_config: Float8MMConfig,
) -> Tuple[Tensor, Tensor]:
    """Preprocess the inner fp8 data tensors for admmm
    Args:
        a_data: Input tensor A.
        b_data: Input tensor B.
        scaled_mm_config: Configuration for _scaled_mm.
    Returns:
        Preprocessed tensors A and B in the format for _scaled_mm.
    """
    if scaled_mm_config.pad_inner_dim:
        assert a_data.size(1) == b_data.size(0), (
            f"Inner dims must match for mm, got {a_data.size(1)} and {b_data.size(0)}"
        )
        a_data = pad_tensor_for_matmul(a_data, dims=1)
        b_data = pad_tensor_for_matmul(b_data, dims=0)
    if not is_row_major(a_data.stride()):
        a_data = a_data.contiguous()
    if is_row_major(b_data.stride()):
        b_data = b_data.t().contiguous().t()
    return a_data, b_data


def preprocess_scale(input_scale: torch.Tensor, input_shape: Tuple[int, ...]):
    """Ensures input tensor is correctly formatted for _scaled_mm"""

    # For PerTensor quantization, scale should be a scalar or have shape [1]
    if input_scale.numel() == 1:
        # Already a scalar, ensure it has the right shape for _scaled_mm
        return input_scale.reshape(1, 1)

    # For per-row/block quantization, we need to handle the reshaping
    input_scale = input_scale.unsqueeze(-1)

    # Match: #input_data.reshape(-1, input_data.shape[-1])
    if input_scale.dim() > 2:
        input_scale = input_scale.reshape(-1, input_scale.shape[-1])

    return input_scale


def addmm_float8_unwrapped_inference(
    a_data: Tensor,
    a_scale: Tensor,
    b_data: Tensor,
    b_scale: Tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    """
    This is the unwrapped version of addmm_float8, which does not take in Float8TrainingTensors
    as inputs. This is used to standardize the logic between subclassed and non subclassed
    versions of the linear module.
    """

    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_scale,
            scale_b=b_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        return output + bias
    return torch._scaled_mm(
        a_data,
        b_data,
        scale_a=a_scale,
        scale_b=b_scale,
        bias=bias,
        scale_result=output_scale,
        out_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )


def _slice_scale_for_dimension(
    scale: torch.Tensor,
    data_shape: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
) -> torch.Tensor:
    """
    Slice the scale tensor appropriately based on the data tensor slicing.
    This function calculates how the scale should be sliced when the data tensor
    is sliced along a given dimension, taking into account the block structure.
    """
    aten = torch.ops.aten

    # Case 1: Per-tensor quantization (scalar scale)
    if scale.numel() <= 1:
        return scale

    # Case 2: Per-row quantization (1D scale)
    # Scale is per-element along this dimension
    if scale.ndim == 1:
        if dim == 0:
            return aten.slice.Tensor(scale, 0, start, end, step)
        else:
            return scale

    # Case 3: Per-block quantization (2D scale)
    block_sizes = tuple(
        data_shape[i] // scale.shape[i] for i in range(len(scale.shape))
    )

    block_size_for_dim = block_sizes[dim]

    if step > 1:
        raise NotImplementedError(
            "Slicing with step > 1 is not implemented for scale tensors."
        )

    # There is blocking in this dimension
    # Calculate which scale elements correspond to the sliced data
    scale_start = start // block_size_for_dim if start is not None else None
    scale_end = (
        (end + block_size_for_dim - 1) // block_size_for_dim
        if end is not None
        else None
    )

    return aten.slice.Tensor(scale, dim, scale_start, scale_end, 1)


def _is_rowwise_scaled(x: torch.Tensor) -> bool:
    """Checks if a quantized tensor is rowwise scaled
    Args:
        x: quantized tensor (should have `block_size` attribute)
    """
    assert hasattr(x, "block_size"), "Expecting input to have `block_size` attribute"
    return tuple(x.block_size) == (1,) * (x.dim() - 1) + (x.shape[-1],)


def _is_tensorwise_scaled(x: torch.Tensor) -> bool:
    """Checks if a quantized tensor is rowwise scaled
    Args:
        x: quantized tensor (should have `block_size` attribute)
    """
    assert hasattr(x, "block_size"), "Expecting input to have `block_size` attribute"
    return all(
        x.block_size[i] == -1 or x.block_size[i] == x.shape[i] for i in range(x.ndim)
    )


def _is_1_128_scaled(x: torch.Tensor) -> bool:
    """Checks if a quantized tensor is scaled with a block size of 1x128
    Args:
        x: quantized tensor (should have `block_size` attribute)
    """
    assert hasattr(x, "block_size"), "Expecting input to have `block_size` attribute"
    b = x.block_size
    return len(b) >= 2 and math.prod(b[:-1]) == 1 and b[-1] == 128


def _is_128_128_scaled(x: torch.Tensor) -> bool:
    """Checks if a quantized tensor is scaled with a block size of 128x128
    Args:
        x: quantized tensor (should have `block_size` attribute)
    """
    assert hasattr(x, "block_size"), "Expecting input to have `block_size` attribute"
    b = x.block_size
    return len(b) == 2 and b[0] == 128 and b[1] == 128


def _granularity_is_a_1_128_w_128_128(
    g: Union[
        FP8Granularity,
        Tuple[FP8Granularity, FP8Granularity],
        list[FP8Granularity],
    ],
) -> bool:
    return len(g) == 2 and g[0] == PerBlock([1, 128]) and g[1] == PerBlock([128, 128])


def _normalize_granularity(
    granularity: Optional[
        Union[
            FP8Granularity,
            Tuple[FP8Granularity, FP8Granularity],
            list[FP8Granularity],
        ]
    ],
) -> Tuple[FP8Granularity, FP8Granularity]:
    processed_granularity = None
    if granularity is None:
        processed_granularity = (PerTensor(), PerTensor())
    elif isinstance(granularity, (PerTensor, PerRow)):
        processed_granularity = (granularity, granularity)
    elif isinstance(granularity, (tuple, list)) and len(granularity) == 2:
        is_per_tensor = isinstance(granularity[0], PerTensor) and isinstance(
            granularity[1], PerTensor
        )
        is_per_row = isinstance(granularity[0], PerRow) and isinstance(
            granularity[1], PerRow
        )
        is_a_1_128_w_128_128 = _granularity_is_a_1_128_w_128_128(granularity)

        if not (is_per_tensor or is_per_row or is_a_1_128_w_128_128):
            raise ValueError(f"Unsupported granularity types: {granularity}.")
        if not isinstance(granularity[0], type(granularity[1])):
            raise ValueError(
                f"Different granularities for activation and weight are not supported: {granularity}."
            )
        processed_granularity = tuple(granularity)
    else:
        raise ValueError(f"Invalid granularity specification: {granularity}.")
    return processed_granularity


def _check_hardware_support(
    granularities: Tuple[FP8Granularity, FP8Granularity],
) -> None:
    """
    Validate that the hardware supports the requested granularities.

    Args:
        granularities: Tuple of (activation_granularity, weight_granularity)

    Raises:
        AssertionError: If hardware doesn't support the requested granularity
        ValueError: If invalid granularity type is provided
    """
    is_per_tensor = isinstance(granularities[0], PerTensor) and isinstance(
        granularities[1], PerTensor
    )
    is_per_row = isinstance(granularities[0], PerRow) and isinstance(
        granularities[1], PerRow
    )
    is_a_1_128_w_128_128 = _granularity_is_a_1_128_w_128_128(granularities)

    if is_per_tensor or is_per_row:
        assert is_sm_at_least_89() or is_MI300(), (
            "Float8 dynamic quantization requires CUDA compute capability ≥8.9 or MI300+."
        )
    elif is_a_1_128_w_128_128:
        # TODO(future PR): look into AMD support
        assert is_sm_at_least_89(), (
            "Float8 1x128 activation and 128x128 weight scaling requires CUDA compute capability ≥8.9."
        )
    else:
        raise ValueError(f"Invalid granularities {granularities}.")
