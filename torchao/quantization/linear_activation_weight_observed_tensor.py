# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Dict, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.observer import AffineQuantizedObserverBase
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "LinearActivationWeightObservedTensor",
]

aten = torch.ops.aten
Tensor = torch.Tensor


class LinearActivationWeightObservedTensor(TorchAOBaseTensor):
    """
    This subclass of Tensor is used in conjuction with a static calibration flow.
    The flow is broken up into 3 parts;
        1. Insert the LinearActivationWeightObservedTensor subclass into the model's nn.Linear layers
        2. Run the model with a calibration dataset, the observer will record the min/max of the input and weight
        3. quantize_ the model to static using the statistics recorded by the observer

    This subclass wraps the original weight tensor on the nn.Linear layer. When forward is called, the observer
    will first calculat statistics on BOTH the input and weight, and then run the linear op.
    """

    original_weight_tensor: torch.Tensor
    input_observer: Optional[AffineQuantizedObserverBase]
    weight_observer: Optional[AffineQuantizedObserverBase]

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_observer: Optional[AffineQuantizedObserverBase] = None,
        weight_observer: Optional[AffineQuantizedObserverBase] = None,
    ):
        kwargs = {}
        dtype = original_weight_tensor.dtype
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        kwargs["device"] = original_weight_tensor.device
        shape = original_weight_tensor.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        input_observer: Optional[AffineQuantizedObserverBase] = None,
        weight_observer: Optional[AffineQuantizedObserverBase] = None,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.input_observer = input_observer
        self.weight_observer = weight_observer

    def __repr__(self):
        return (
            f"LinearActivationWeightObservedTensor(\n"
            f"original_weight={self.original_weight_tensor}\n"
            f"input_observer={self.input_observer.__class__.__name__ if self.input_observer else None}\n"
            f"weight_observer={self.weight_observer.__class__.__name__ if self.weight_observer else None}\n)"
        )

    def __tensor_flatten__(self):
        return ["original_weight_tensor"], [self.input_observer, self.weight_observer]

    @classmethod
    def __tensor_unflatten__(
        cls,
        tensor_data_dict: Dict[str, Tensor],
        tensor_attributes,
        outer_size,
        outer_stride,
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        (input_observer, weight_observer) = tensor_attributes
        return cls(original_weight_tensor, input_observer, weight_observer)

    @classmethod
    def from_float(
        cls,
        original_weight_tensor: Tensor,
        input_observer: Optional[AffineQuantizedObserverBase] = None,
        weight_observer: Optional[AffineQuantizedObserverBase] = None,
    ):
        return cls(original_weight_tensor, input_observer, weight_observer)

    def _apply_fn_to_data(self, fn: Callable):
        """Applies a fn to the tensor component of the LinearActivationWeightObservedTensor"""
        return self.__class__(
            fn(self.original_weight_tensor),
            self.input_observer,
            self.weight_observer,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self._apply_fn_to_data(lambda x: x.to(**kwargs))


implements = LinearActivationWeightObservedTensor.implements
implements_torch_function = (
    LinearActivationWeightObservedTensor.implements_torch_function
)


@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if weight_tensor.input_observer is not None:
        input_tensor = weight_tensor.input_observer(input_tensor)
    if weight_tensor.weight_observer is not None:
        weight_tensor = weight_tensor.weight_observer(
            weight_tensor.original_weight_tensor
        )
    else:
        weight_tensor = weight_tensor.original_weight_tensor

    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )


# Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([LinearActivationWeightObservedTensor])
