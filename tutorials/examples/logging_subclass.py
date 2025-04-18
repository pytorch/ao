# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.utils._pytree as pytree


class LoggingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            a.shape,
            strides=a.stride(),
            storage_offset=a.storage_offset(),
            dtype=a.dtype,
            device=a.device,
        )

    def __init__(self, a):
        self.a = a

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        print("func: " + str(func))
        # Our logging subclass trivially implements *every* pytorch op.
        # It does so by:
        # - unwrapping any LoggingTensor arguments
        # - calling the underlying function on the inner tensors
        # - wrapping any tensor outputs into LoggingTensors
        args_a = pytree.tree_map_only(LoggingTensor, lambda x: x.a, args)
        kwargs_a = pytree.tree_map_only(LoggingTensor, lambda x: x.a, kwargs)
        out_a = func(*args_a, **kwargs_a)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_flat = [
            cls(o_a) if isinstance(o_a, torch.Tensor) else o_a for o_a in out_a_flat
        ]
        return pytree.tree_unflatten(out_flat, spec)


class ToyModel(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    # Set up toy model
    float_model = ToyModel(64, 128, 32).cuda()

    # Replace any linear layer weights with our LoggingTensor
    for name, child in float_model.named_children():
        if type(child) == torch.nn.Linear:
            child.weight = torch.nn.Parameter(
                LoggingTensor(child.weight), requires_grad=True
            )

    # run the model
    with torch.no_grad():
        x = torch.randn(64, 64, 64, device="cuda")
        _ = float_model(x)
