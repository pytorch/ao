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
    tensor_attribute_names = ["observed_data"]

    def __new__(cls, hp_data: torch.Tensor, observed_data):
        shape = hp_data.shape
        kwargs = {}
        kwargs["device"] = hp_data.device
        kwargs["dtype"] = hp_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, hp_data: torch.Tensor, observed_data):
        super().__init__()
        self.hp_data = hp_data
        self.observed_data = observed_data

    def update(self, input: torch.Tensor):
        """Store observation in external registry"""
        self.observed_data.append(input.detach().cpu())

    @classmethod
    def from_hp(cls, hp_tensor):
        return ObserverTensor(hp_tensor, [])


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
