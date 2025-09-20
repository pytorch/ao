# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Optional, Tuple

import torch

from torchao.quantization.quant_primitives import _fake_quantize_affine

from .granularity import (
    Granularity,
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
)
from .quant_primitives import (
    MappingType,
    ZeroPointDomain,
    _get_reduction_params,
    choose_qparams_affine_with_min_max,
)

logger = logging.getLogger(__name__)


# borrowed from torch.ao.quantization.observer
class _PartialWrapper:
    def __init__(self, p):
        self.p = p

    def __call__(self, *args, **keywords):
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__()

    def with_args(self, *args, **kwargs):
        return _with_args(self, *args, **kwargs)


def _with_args(cls_or_self, *args, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, *args, **kwargs))
    return r


def get_block_size(
    input_shape: Tuple[int, ...], granularity: Granularity
) -> Tuple[int, ...]:
    """Get the block size based on the input shape and granularity type.

    Args:
        input_shape: The input tensor shape possibly more than 2 dimensions
        granularity: The granularity type of the quantization
    """
    if isinstance(granularity, PerTensor):
        return input_shape
    elif isinstance(granularity, PerAxis):
        block_size = list(input_shape)
        block_size[granularity.axis] = 1
        return tuple(block_size)
    elif isinstance(granularity, (PerRow, PerToken)):
        return (1,) * (len(input_shape) - 1) + (input_shape[-1],)
    elif isinstance(granularity, PerGroup):
        assert input_shape[-1] % granularity.group_size == 0, (
            f"Group size {granularity.group_size} does not divide input shape {input_shape}"
        )
        return (1,) * (len(input_shape) - 1) + (granularity.group_size,)
    raise ValueError(f"Unsupported Granularity: {granularity}")


ABC: Any = ABCMeta("ABC", (object,), {})  # compatible with Python 2 *and* 3:


class AffineQuantizedObserverBase(ABC, torch.nn.Module):
    """Observer module for affine quantization (https://github.com/pytorch/ao/tree/main/torchao/quantization#affine-quantization)

    Args:
      `granularity` and `block_size`: The granularity of the quantization,
        must specify at least one, if both are specified `block_size` takes precedence
        Current supported granularity type are `PerTensor` and `PerAxis`
      other args: please see `:class:torchao.dtypes.AffineQuantizedTensor`
    """

    with_args = classmethod(_with_args)

    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    ):
        super().__init__()
        assert granularity is not None, "granularity is None"
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")
        self.mapping_type = mapping_type
        self.target_dtype = target_dtype
        self.granularity = granularity
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.scale_dtype = scale_dtype
        self.zero_point_dtype = zero_point_dtype
        self.preserve_zero = preserve_zero
        self.zero_point_domain = zero_point_domain

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward function should take the input tensor
        and updates internal stats and return the original input Tensor
        """
        pass

    @abstractmethod
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameter based on the stats attached to the observer module
        and returns a tuple of scale and zero_point Tensor
        """
        pass


class AffineQuantizedMinMaxObserver(AffineQuantizedObserverBase):
    def forward(self, input: torch.Tensor):
        if input.numel() == 0:
            return input

        input_detached = input.detach()
        assert self.granularity is not None, "granularity is None"
        block_size = get_block_size(input_detached.shape, self.granularity)

        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, input_detached.size()
        )
        input_detached = input_detached.view(shape_for_reduction)
        min_val = torch.amin(input_detached, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input_detached, dim=reduction_dims, keepdim=False)
        if not hasattr(self, "min_val") or not hasattr(self, "max_val"):
            self.min_val = min_val
            self.max_val = max_val
        else:
            assert self.min_val.shape == min_val.shape, (
                f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            )
            assert self.max_val.shape == max_val.shape, (
                f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            )
            min_val = torch.min(self.min_val, min_val)
            max_val = torch.max(self.max_val, max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        # returning original input
        return input

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hasattr(self, "min_val") and hasattr(self, "max_val"), (
            "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        )
        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [],  # BlockSize is not needed because the min/max are already reduced
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )


class AffineQuantizedFixedQParamObserver(AffineQuantizedObserverBase):
    """
    Observer that allows manual setting of fixed quantization parameters.
    """

    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            mapping_type,
            target_dtype,
            granularity,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
        )
        if not scale:
            scale = torch.Tensor([1])
        if not zero_point:
            zero_point = torch.zeros_like(scale)
        self.register_buffer("scale", scale.to(dtype=scale_dtype))
        self.register_buffer("zero_point", zero_point.to(dtype=zero_point_dtype))

    def set_qparams(self, scale, zero_point=None):
        if not zero_point:
            zero_point = torch.zeros_like(scale)
        self.scale = scale.to(dtype=self.scale_dtype)
        self.zero_point = zero_point.to(dtype=self.zero_point_dtype)

    def forward(self, input):
        return input

    def calculate_qparams(self):
        return self.scale, self.zero_point


class AffineQuantizedMSEObserver(AffineQuantizedObserverBase):
    """
    Minimize quantization loss caused by outlier via linear search. More details can be found at https://arxiv.org/pdf/2209.13325
    """

    def __init__(
        self,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        granularity: Granularity,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        steps: int = 100,
        run_once: bool = False,
    ):
        super().__init__(
            mapping_type,
            target_dtype,
            granularity,
            quant_min,
            quant_max,
            eps,
            scale_dtype,
            zero_point_dtype,
            preserve_zero,
            zero_point_domain,
        )
        self.steps = steps
        self.calibrated = False
        self.run_once = run_once

    def mse(self, pred, expect, block_size):
        loss = (pred - expect).abs().pow(2)
        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, loss.size()
        )
        loss = loss.view(shape_for_reduction)
        return torch.mean(loss, dim=reduction_dims, keepdim=False)

    def loss_fn(self, x, new_min, new_max):
        block_size = get_block_size(x.shape, self.granularity)
        scale, zero_point = choose_qparams_affine_with_min_max(
            new_min,
            new_max,
            self.mapping_type,
            [],
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )
        x_q = _fake_quantize_affine(
            x,
            block_size,
            scale,
            zero_point,
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
        )
        return self.mse(x_q, x, block_size)

    def line_search(self, input):
        if input.numel() == 0:
            return input

        input_detached = input.detach()
        assert self.granularity is not None, "granularity is None"
        block_size = get_block_size(input_detached.shape, self.granularity)

        shape_for_reduction, reduction_dims = _get_reduction_params(
            block_size, input_detached.size()
        )
        input_detached = input_detached.view(shape_for_reduction)
        min_val = torch.amin(input_detached, dim=reduction_dims, keepdim=False)
        max_val = torch.amax(input_detached, dim=reduction_dims, keepdim=False)

        range_val = torch.max(min_val.abs(), max_val)
        optimal_loss = torch.zeros_like(min_val) + 1e9

        # check which clip range could produce smallest loss
        for i in range(1, self.steps + 1):
            thres = range_val / self.steps * i
            current_loss = self.loss_fn(input, -thres, thres)
            min_val = torch.where(current_loss < optimal_loss, -thres, min_val)
            max_val = torch.where(current_loss < optimal_loss, thres, max_val)
            optimal_loss = torch.min(current_loss, optimal_loss)

        return min_val, max_val

    def forward(self, input):
        if not (self.run_once and self.calibrated):
            self.min_val, self.max_val = self.line_search(input)
            self.calibrated = True

        return input

    def calculate_qparams(self):
        assert hasattr(self, "min_val") and hasattr(self, "max_val"), (
            "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        )
        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [],
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )


# Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([PerRow, PerTensor])
