import torch
from .granularity import (
    Granularity,
    PerAxis,
    PerRow,
    PerTensor,
)
from .quant_primitives import (
    _get_reduction_params,
    choose_qparams_affine_with_min_max,
    MappingType,
    ZeroPointDomain,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Any
from functools import partial
import logging

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
    elif isinstance(granularity, PerRow):
        return (1,) * (len(input_shape) - 1) + (input_shape[-1],)
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
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
    ):
        super().__init__()
        assert granularity is not None, "granularity is None"

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
            assert self.min_val.shape == min_val.shape, f"Can't update existing min_val - shape mismatch, self.min_val:{self.min_val.shape} != min_val:{min_val.shape}"
            assert self.max_val.shape == max_val.shape, f"Can't update existing max_val - shape mismatch, self.max_val {self.max_val.shape} != max_val:{max_val.shape}"
            min_val = torch.min(self.min_val, min_val)
            max_val = torch.max(self.max_val, max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        # returning original input
        return input

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            hasattr(self, "min_val") and hasattr(self, "max_val")
        ), "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        return choose_qparams_affine_with_min_max(
            self.min_val,
            self.max_val,
            self.mapping_type,
            [], # BlockSize is not needed because the min/max are already reduced
            self.target_dtype,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.scale_dtype,
            self.zero_point_dtype,
            self.preserve_zero,
            self.zero_point_domain,
        )

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([PerRow, PerTensor])
