# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_reduce

from torchao.float8.config import ScalingGranularity

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


@torch.no_grad()
def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: The float8 dtype.
        round_scales_to_power_of_2: if true, round scaling factor down to the nearest power of 2.
    """
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    amax = amax.to(torch.float64)
    if float8_dtype in FP8_TYPES:
        res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=EPS)
        res = res.to(torch.float32)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")
    if round_scales_to_power_of_2:
        res = _round_scale_down_to_power_of_2(res)
    return res


@torch.no_grad()
def tensor_to_amax(
    x: torch.Tensor,
    reduce_amax: bool = False,
    device_mesh=None,
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    axiswise_dim: Optional[int] = None,
) -> torch.Tensor:
    if scaling_granularity is ScalingGranularity.TENSORWISE:
        amax = torch.max(torch.abs(x))
    else:
        assert scaling_granularity is ScalingGranularity.AXISWISE, "unsupported"
        assert axiswise_dim is not None, "unsupported"
        amax = torch.amax(torch.abs(x), dim=axiswise_dim, keepdim=True)

    # If the user asked for distributed reduction, do it.
    # If the user did not ask for it, assume that it will
    # happen elsewhere.
    if reduce_amax and dist.is_initialized():
        pg = device_mesh.get_group() if device_mesh is not None else None
        # dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=pg)
        group = list(range(dist.get_world_size())) if pg is None else pg
        amax = all_reduce(amax, "MAX", group)
        if isinstance(amax, AsyncCollectiveTensor):
            amax = amax.wait()

    return amax


@torch.no_grad()
def tensor_to_scale(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    reduce_amax: bool = False,
    device_mesh=None,
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    axiswise_dim: Optional[int] = None,
    round_scales_to_power_of_2: bool = False,
) -> torch.Tensor:
    """
    Compute scaling factor for the given high precision tensor.

    Args:
        hp_tensor: high precision tensor
        float8_dtype: the float8 dtype to use
        reduce_amax: whether to reduce the max(abs(hp_tensor)) value across distributed ranks
        scaling_granularity: Defines the scaling granularity
        axiswise_dim: if axiswise granularity is used, defines the dim to scale across
        round_scales_to_power_of_2: if true, round scaling factor down to the nearest power of 2.
    """
    amax = tensor_to_amax(
        hp_tensor,
        reduce_amax,
        device_mesh,
        scaling_granularity,
        axiswise_dim,
    )
    return amax_to_scale(
        amax, float8_dtype, round_scales_to_power_of_2=round_scales_to_power_of_2
    )


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
    tensor: torch.Tensor, float8_dtype: torch.dtype
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


def _is_row_major(stride):
    assert len(stride) == 2, "_is_row_major only supports 2D tensors"
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
    return (1 + ((size - 1) // alignment_value)) * alignment_value


def _pad_tensor_for_matmul(
    tensor: torch.Tensor, dims: Union[int, Iterable[int]]
) -> torch.Tensor:
    """
    Pads a 2D tensor with zeros to ensure that its dimensions are multiples of 16, which is required `torch._scaled_mm`

    Args:
        tensor: The tensor to pad.
        dims: Dimensions to pad.

    Returns:
        torch.Tensor: The padded tensor.

    Usage:
    ```
        >>> _pad_tensor_for_matmul(torch.randn((10, 10)), dims=0).shape
        torch.Size([16, 10])
        >>> _pad_tensor_for_matmul(torch.randn((10, 10)), dims=1).shape
        torch.Size([10, 16])
        >>> _pad_tensor_for_matmul(torch.randn((10, 10)), dims=(0, 1)).shape
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

    # Calculate padding values for both dimensions
    pad_dim1 = dim1_aligned - dim1
    pad_dim2 = dim2_aligned - dim2

    return torch.nn.functional.pad(tensor, (0, pad_dim2, 0, pad_dim1))


# for BC, confirmed there are users using this util function
pad_tensor_for_matmul = _pad_tensor_for_matmul


def _round_scale_down_to_power_of_2(scale: torch.Tensor):
    assert scale.dtype == torch.float32, "scale must be float32 tensor"
    return torch.exp2(torch.floor(torch.log2(scale)))
