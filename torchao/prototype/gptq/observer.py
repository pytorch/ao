# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchao.utils import TorchAOBaseTensor


class ObserverTensor(TorchAOBaseTensor):
    tensor_data_names = ["hp_data"]
    optional_tensor_data_names = ["hessian"]
    tensor_attribute_names = ["total_batches"]

    def __new__(cls, hp_data: torch.Tensor, total_batches: int, hessian=None):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, hp_data: torch.Tensor, total_batches: int, hessian=None):
        super().__init__()
        self.hp_data = hp_data
        self.hessian = hessian
        self.total_batches = total_batches

    def update(self, input: torch.Tensor):
        # if torch.isnan(input).any():
        #     print("nan in input")
        #     breakpoint()
        """Incrementally update Hessian matrix from input activations."""
        # Move input to same device as hp_data and convert to float
        x = input.float().to(self.hp_data.device)
        shape = x.shape

        # Calculate batch size
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

        # Update Hessian: x = ((2 / total_batches) ** (1 / 2)) * x.t()
        x = ((2 / self.total_batches) ** (1 / 2)) * x.t()
        self.hessian += x.matmul(x.t())

    @classmethod
    def from_hp(cls, hp_tensor):
        return ObserverTensor(hp_tensor, 0, None)


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
            f"Expected weight_tensor to be ObserverTensor, got: {type(weight_tensor)}"
        )


@implements(aten.bmm.default)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor.update(input_tensor.detach())
    return func(input_tensor, weight_tensor.hp_data)
