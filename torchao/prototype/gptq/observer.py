# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.nn.functional as F

from torchao.utils import TorchAOBaseTensor


class ObserverTensor(TorchAOBaseTensor):
    """Wraps a weight tensor to observe input activations during forward passes.

    Optionally accepts a quantize_fn that is applied to each input before saving,
    so the Hessian reflects quantized activation noise at inference time (e.g. for
    dynamic quantization schemes like MXFP8/MXFP4).
    """

    tensor_data_names = ["hp_data"]
    tensor_attribute_names = ["observed_inputs"]

    def __new__(
        cls,
        hp_data: torch.Tensor,
        observed_inputs: List[torch.Tensor],
        quantize_fn=None,
    ):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        hp_data: torch.Tensor,
        observed_inputs: List[torch.Tensor],
        quantize_fn=None,
    ):
        super().__init__()
        self.hp_data = hp_data
        self.observed_inputs = observed_inputs
        self.quantize_fn = quantize_fn

    def update(self, input: torch.Tensor):
        if self.quantize_fn is not None:
            input = self.quantize_fn(input)
        self.observed_inputs.append(input.cpu())

    @classmethod
    def from_hp(cls, hp_tensor: torch.Tensor, quantize_fn=None):
        return cls(hp_tensor, [], quantize_fn)


implements = ObserverTensor.implements
implements_torch_function = ObserverTensor.implements_torch_function
aten = torch.ops.aten


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, ObserverTensor):
        weight_tensor.update(input_tensor.detach())
        return F.linear(input_tensor, weight_tensor.hp_data, bias)
    else:
        raise ValueError(
            f"Expected weight_tensor to be GPTQObserverTensor, got: {type(weight_tensor)}"
        )


@implements(aten.bmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor.update(input_tensor.detach())
    return func(input_tensor, weight_tensor.hp_data)


class GPTQObserverTensor(ObserverTensor):
    """ObserverTensor that incrementally computes the Hessian during observation.

    Unlike the base ObserverTensor which stores all inputs and computes the Hessian
    at convert time, this computes the Hessian on-the-fly during forward passes,
    avoiding the need to store all activations in memory.
    """

    tensor_data_names = ["hp_data"]
    optional_tensor_data_names = ["hessian"]
    tensor_attribute_names = ["total_batches"]

    def __new__(
        cls, hp_data: torch.Tensor, total_batches: int, hessian=None, quantize_fn=None
    ):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self, hp_data: torch.Tensor, total_batches: int, hessian=None, quantize_fn=None
    ):
        super(ObserverTensor).__init__()
        self.hp_data = hp_data
        self.hessian = hessian
        self.total_batches = total_batches
        self.quantize_fn = quantize_fn

    def update(self, input: torch.Tensor):
        """Incrementally update Hessian matrix from input activations."""
        if self.quantize_fn is not None:
            input = self.quantize_fn(input)

        x = input.float().to(self.hp_data.device)
        shape = x.shape

        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Lazily initialize Hessian on first call
        if self.hessian is None:
            feature_dim = x.shape[-1]
            self.hessian = torch.zeros(
                feature_dim,
                feature_dim,
                dtype=torch.float32,
                device=self.hp_data.device,
            )

        # Apply running average formula
        if self.total_batches > 0:
            self.hessian *= self.total_batches / (self.total_batches + n)

        self.total_batches += n

        x = ((2 / self.total_batches) ** (1 / 2)) * x.t()
        self.hessian += x.matmul(x.t())

    @classmethod
    def from_hp(cls, hp_tensor, quantize_fn=None):
        return GPTQObserverTensor(hp_tensor, 0, None, quantize_fn)


def _calculate_hessian(inputs, device=None, quantize_fn=None):
    """Calculate Hessian matrix from input activations for GPTQ.

    Args:
        inputs: List of input activation tensors.
        device: Device to move inputs to before computation.
        quantize_fn: Optional callable (Tensor -> Tensor) that quantizes and
            dequantizes each input before Hessian computation, so the Hessian
            reflects activation quantization noise at inference time.
    """
    H = 0
    total_batches = 0

    for inp in inputs:
        # Setup x (activation tensor)
        if quantize_fn is not None:
            inp = quantize_fn(inp)
        x = inp.float()
        if device:
            x = x.to(device)
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Update Hessian with running average
        H *= total_batches / (total_batches + n)
        total_batches += n

        x = ((2 / total_batches) ** (1 / 2)) * x.t()
        H += x.matmul(x.t())

    return H
