# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Literal, Tuple, Union, Optional

import torchao.float8.config as config

import torch
import torch.distributed as dist

# Helpful visualizer for debugging (only supports fp32):
# https://www.h-schmidt.net/FloatConverter/IEEE754.html

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12

IS_ROCM = torch.cuda.is_available() and torch.version.hip is not None
FP8_TYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


# User defined type for using the individual F8 type based on config
e4m3_dtype = torch.float8_e4m3fn if not config.use_fnuz_dtype else torch.float8_e4m3fnuz
e5m2_dtype = torch.float8_e5m2 if not config.use_fnuz_dtype else torch.float8_e5m2fnuz


@torch.no_grad()
def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: The float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    if float8_dtype in FP8_TYPES:
        res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)
    return res.to(torch.float32)


@torch.no_grad()
def amax_history_to_scale(
    amax_history: torch.Tensor,
    float8_dtype: torch.Tensor,
    orig_dtype: torch.dtype,
    history_to_scale_fn_type: Literal["max"],
    stack: bool
):
    """Takes in a history of amax values and returns a scale tensor.
    Args:
        amax_history: A tensor containing the history of amax values.
        float8_dtype: The float8 dtype.
        orig_dtype: The original dtype of the tensor.
        history_to_scale_fn_type: The type of function to use to convert the history to a scale.
        stack: Whether the amax_history is a stack of amax histories and we will calculate amax of each entry in the stack.
    """
    if history_to_scale_fn_type == "max":
        if stack:
            amax = torch.max(amax_history, dim=1).values
        else:
            amax = torch.max(amax_history)
        return amax_to_scale(amax, float8_dtype, orig_dtype)
    raise NotImplementedError(
        f"Invalid history_to_scale_fn_type, only 'max' is supported. Got: {history_to_scale_fn_type}"
    )


@torch.no_grad()
def tensor_to_amax(
    x: torch.Tensor, tile_size: Optional[Tuple[int, int]], reduce_amax: bool = False
) -> torch.Tensor:
    if tile_size is None:
        amax = torch.max(torch.abs(x))
    else:
        assert x.dim(), "NYI; only handles 2d inputs and group sizes for now"
        assert x.dim() == len(
            tile_size
        ), f"len(tile_size) must match tensor dim, got len(tile_size)={len(tile_size)} and x.dim={x.dim()}"
        assert x.shape[0] % tile_size[0] == 0 and x.shape[1] % tile_size[1] == 0, "Tensor shape must be divisible by group size"
        tiled = x.unfold(0, tile_size[0], tile_size[0]).unfold(
            1, tile_size[1], tile_size[1]
        )
        amax = torch.max(torch.abs(tiled), dim=(-1)).values.max(dim=-1).values

    # If the user asked for distributed reduction, do it.
    # If the user did not ask for it, assume that it will
    # happen elsewhere.
    if reduce_amax and dist.is_initialized():
        dist.all_reduce(amax, op=dist.ReduceOp.MAX)

    return amax


@torch.no_grad()
def tensor_to_scale(
    x: torch.Tensor,
    float8_dtype: torch.dtype,
    tile_size: Optional[Tuple[int, int]],
    reduce_amax: bool = False,
) -> torch.Tensor:
    amax = tensor_to_amax(x, tile_size, reduce_amax=reduce_amax)
    return amax_to_scale(amax, float8_dtype, x.dtype)


def to_fp8_saturated(x: torch.Tensor, float8_dtype: torch.dtype):
    """Converts a tensor to a saturated fp8 tensor.

    Note:
        The default behavior in PyTorch for casting to `float8_e4m3fn`
        and `e5m2` is to not saturate. In this context, we should saturate.
        A common case where we want to saturate is when the history of a
        tensor has a maximum value of `amax1`, and the current amax value
        is `amax2`, where `amax1 < amax2`. This is common when using delayed
        scaling.
    """
    if float8_dtype in FP8_TYPES:
        max_value = torch.finfo(float8_dtype).max
        x = x.clamp(min=-max_value, max=max_value)
        return x.to(float8_dtype)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")


def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the error between two tensors in dB.

    For more details see:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        x: The original tensor.
        y: The tensor to compare to the original tensor.
    """
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


def fp8_tensor_statistics(
    tensor: torch.Tensor, float8_dtype=e4m3_dtype
) -> Tuple[int, ...]:
    """Calculate FP8 tensor stats

    Args:
        tensor: The tensor to calculate stats for.
        float8_dtype: The float8 dtype.

    Returns:
        A tuple containing the number of zeros and the number of max values.
    """
    if float8_dtype in FP8_TYPES:
        FP8_MAX = torch.finfo(float8_dtype).max
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")
    tensor_orig_type = tensor._data.to(dtype=tensor._orig_dtype)
    num_max = (torch.abs(tensor_orig_type) == FP8_MAX).sum().item()
    num_zero = (tensor_orig_type == 0).sum().item()
    return (num_zero, num_max)


def is_row_major(stride):
    assert len(stride) == 2, "is_row_major only supports 2D tensors"
    return stride[0] > stride[1] and stride[1] == 1


def _get_min_alignment(size: int, alignment_value: int) -> int:
    """
    Returns the minimum alignment value that is greater than or equal to the given size.

    Args:
        size: The size of the data to be aligned.
        alignment_value: The alignment value to be used.

    Returns:
        int: The minimum alignment value that is greater than or equal to the given size.

    Usage:
    ```
        >>> _get_min_alignment(10, 8)
        16
    ```
    """
    if size % alignment_value == 0:
        return size
    return (1 + (size // alignment_value)) * alignment_value


def pad_tensor_for_matmul(
    tensor: torch.Tensor, dims: Union[int, Iterable[int]]
) -> torch.Tensor:
    """
    Pads a 2D tensor with zeros to ensure that its dimensions are multiples of 16, which is required `torch._scaled_mm`

    Args:
        tensor: The tensor to pad.
        both: Whether to pad both dimensions or just the second dimension.

    Returns:
        torch.Tensor: The padded tensor.

    Usage:
    ```
        >>> pad_tensor_for_matmul(torch.randn((10, 10)), dims=0).shape
        torch.Size([16, 10])
        >>> pad_tensor_for_matmul(torch.randn((10, 10)), dims=1).shape
        torch.Size([10, 16])
        >>> pad_tensor_for_matmul(torch.randn((10, 10)), dims=(0, 1)).shape
        torch.Size([16, 16])
    ```
    """
    assert tensor.dim() == 2
    dim1, dim2 = tensor.shape

    if isinstance(dims, int):
        dims = (dims,)

    # Calculate aligned dimensions based on the specified dims
    dim1_aligned = _get_min_alignment(dim1, 16) if 0 in dims else dim1
    dim2_aligned = _get_min_alignment(dim2, 16) if 1 in dims else dim2

    # Check if padding is needed for either dimension
    if dim1 == dim1_aligned and dim2 == dim2_aligned:
        return tensor

    # Calculate padding values for both dimensions
    pad_dim1 = dim1_aligned - dim1
    pad_dim2 = dim2_aligned - dim2

    return torch.nn.functional.pad(tensor, (0, pad_dim2, 0, pad_dim1))


def repeat_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Repeat the scale tensor to match the dimensions of the input tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        scale (torch.Tensor): The scale tensor to be repeated.
    
    Returns:
        torch.Tensor: The repeated scale tensor.
    
    Raises:
        ValueError: If the scale and tensor dimensions are incompatible.
    """
    # Check dimensions
    if not (scale.dim() in {0, 1} or (scale.dim() == tensor.dim() and scale.dim() == 2)):
        raise ValueError(f"Scale and tensor must have compatible dimensions. "
                         f"Got scale.dim() = {scale.dim()} and tensor.dim() = {tensor.dim()}")
    
    # Check if scale is a scalar (0-dim tensor)
    if scale.dim() <= 1:
        return scale
    
    # Initialize repeated scale
    scales_repeated = scale
    
    # Repeat scale if necessary
    if scale.dim() > 1:  # Skip this part if scale is 1-dimensional
        for i in range(tensor.dim()):
            # Check if repetition is needed
            if tensor.shape[i] // scale.shape[i] not in {tensor.shape[i], 1}:
                repeat_factor = tensor.shape[i] // scale.shape[i]
                scales_repeated = scales_repeated.repeat_interleave(repeat_factor, dim=i)
    
    return scales_repeated
