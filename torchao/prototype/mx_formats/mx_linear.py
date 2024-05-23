# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the UX for converting a model to use mx weights

For now, this is a module swap for speed of iteration.

Eventually we plan to move this to a tensor subclass weight wrapper for
inference, and to a tensor subclass weight wrapper + module hooks for training.
"""

import torch
import torch.nn.functional as F

from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_mx


@torch._dynamo.allow_in_graph
class NoopFwToMXBw(torch.autograd.Function):
    """
    Forward: no-op
    Backward: cast grad to MX
    """

    @staticmethod
    def forward(ctx, x, elem_dtype, block_size):
        ctx.elem_dtype = elem_dtype
        ctx.block_size = block_size
        return x

    @staticmethod
    def backward(ctx, g):
        scale, data = to_mx(g, ctx.elem_dtype, ctx.block_size)
        return (
            MXTensor(scale, data, ctx.elem_dtype, ctx.block_size, g.dtype),
            None,
            None,
        )


class MXLinear(torch.nn.Linear):
    """
    Linear layer with the compute happening in emulate MX. Currently the MX
    matmul is emulated since there is no hardware support yet. Activations,
    weights and grads are casted to MX and back to high precision for each
    matmul.
    """

    @classmethod
    @torch.no_grad()
    def from_float(cls, mod, elem_dtype, block_size):
        mod.__class__ = MXLinear
        mod.elem_dtype = elem_dtype
        mod.block_size = block_size
        return mod

    def forward(self, x):
        x_mx = MXTensor.to_mx(x, self.elem_dtype, self.block_size)
        w_mx = MXTensor.to_mx(self.weight, self.elem_dtype, self.block_size)
        y = F.linear(x_mx, w_mx, self.bias)
        y = NoopFwToMXBw.apply(y, self.elem_dtype, self.block_size)
        return y


class MXInferenceLinear(torch.nn.Linear):
    """
    Inference version of MXLinear, with the weight pre-quantized to MX.
    """

    @classmethod
    @torch.no_grad()
    def from_float(cls, mod, elem_dtype, block_size):
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(**super_kwargs)
        # TODO(future PR): set to new_mod.weight directly, will need to work
        # through some errors
        new_mod.weight_mx = MXTensor.to_mx(
            mod.weight.t().contiguous(), elem_dtype, block_size=block_size
        ).t()
        new_mod.bias = mod.bias
        new_mod.elem_dtype = elem_dtype
        return new_mod

    @torch.no_grad()
    def forward(self, x):
        w_hp = self.weight_mx.to_dtype(x.dtype)
        y = F.linear(x, w_hp, self.bias)
        return y


def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=""
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn
            )


def _is_linear(mod, fqn):
    return isinstance(mod, torch.nn.Linear)


def swap_linear_with_mx_linear(model, elem_dtype, block_size, filter_fn=None):
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXLinear.from_float(mod, elem_dtype, block_size),
        combined_filter_fn,
    )


def swap_linear_with_mx_inference_linear(
    model,
    elem_dtype,
    block_size,
    filter_fn=None,
):
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXInferenceLinear.from_float(mod, elem_dtype, block_size),
        combined_filter_fn,
    )
