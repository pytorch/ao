# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "ObservedLinear",
]

aten = torch.ops.aten


# TODO: Build enum backend for more observers like AWQObserver, PerChannelHistogramObserver
# User should be able to handle `act_obs` directly for easy customization
class ObservedLinear(TorchAOBaseTensor):
    """
    A tensor subclass for static quantization with activation observation.

    This subclass wraps the weight tensor and adds an observer that collects
    activation statistics during calibration. After calibration, use `quantize_`
    with subclass to convert to a quantized module.

    Example usage:
        # Step 1: PREPARE - Insert observers into the model
        model = torch.nn.Sequential(torch.nn.Linear(64, 128))
        quantize_(model, Int8StaticActivationInt8WeightConfig(step="prepare", granularity=PerRow()))

        # Step 2: CALIBRATE - Run calibration data to collect activation statistics
        for _ in range(10):
            calibration_input = torch.randn(32, 64)
            model(calibration_input)

        # Step 3: CONVERT - Convert observers to quantized layers
        quantize_(model, Int8StaticActivationInt8WeightConfig(step="convert"))

        # Step 4: INFERENCE - Use the quantized model
        output = model(torch.randn(32, 64))
    """

    tensor_data_names = ["original_weight_tensor"]
    tensor_attribute_names = ["act_obs"]

    original_weight_tensor: torch.Tensor
    act_obs: AffineQuantizedMinMaxObserver

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        act_obs: AffineQuantizedMinMaxObserver,
    ):
        kwargs = {
            "dtype": original_weight_tensor.dtype,
            "requires_grad": False,
            "device": original_weight_tensor.device,
        }
        return torch.Tensor._make_wrapper_subclass(
            cls, original_weight_tensor.shape, **kwargs
        )

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        act_obs: AffineQuantizedMinMaxObserver,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.act_obs = act_obs

    def _apply_fn_to_data(self, fn):
        """Applies a fn to the tensor component of the ObservedLinear"""
        return self.__class__(
            fn(self.original_weight_tensor),
            self.act_obs,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self._apply_fn_to_data(lambda x: x.to(**kwargs))

    @classmethod
    def from_hp(
        cls,
        hp_linear: torch.nn.Linear,
        act_obs: AffineQuantizedMinMaxObserver,
    ) -> "ObservedLinear":
        """Create an observed linear tensor from a high-precision linear module."""
        return cls(hp_linear.weight, act_obs)


implements = ObservedLinear.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    # Observe the input activation
    weight_tensor.act_obs(input_tensor)
    # Use the original weight tensor
    return torch.nn.functional.linear(
        input_tensor, weight_tensor.original_weight_tensor, bias
    )


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].__class__(
            args[0].original_weight_tensor.detach(),
            args[0].act_obs,
        ),
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].__class__(
            args[0].original_weight_tensor.clone(),
            args[0].act_obs,
        ),
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].__class__(
            args[0].original_weight_tensor.t(),
            args[0].act_obs,
        ),
    )


@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    bias, input_tensor, weight_tensor = args
    # Observe the input activation
    # addmm: output = Ax (input x weight) + B (bias)
    weight_tensor.act_obs(input_tensor)
    return func(bias, input_tensor, weight_tensor.original_weight_tensor, **kwargs)


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0]._apply_fn_to_data(lambda x: func(x, **kwargs)),
    )


# Allow a model with ObservedLinear weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([ObservedLinear])
