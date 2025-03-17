# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Any, List, Tuple

import torch


def int8_symmetric_quantize(
    fp32_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetrically quantize the torch.float32 tensor into torch.int8.
    Return a 2-tuple of (quantized value, scale).
    """
    quant_min = -128
    quant_max = 127
    min_val = torch.amin(fp32_tensor, dim=[1], keepdim=False)
    max_val = torch.amax(fp32_tensor, dim=[1], keepdim=False)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    scale = scale.view(fp32_tensor.shape[0], -1)
    out = torch.round(fp32_tensor * (1.0 / scale))
    out = torch.clamp(out, quant_min, quant_max).to(torch.int8)
    return out, scale


# Our subclass represents a tensor that has been quantized to int8
# It will hold two inner tensors:
# - int_data: int8[M, N]
# - scale: fp32[M, 1]
class Int8SymmetricTensor(torch.Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: torch.Tensor, scale: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            strides=int_data.stride(),
            storage_offset=int_data.storage_offset(),
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: torch.Tensor, scale: torch.Tensor):
        # inner data expected to be quantized already
        assert int_data.dtype is torch.int8
        # we could do more work to support ndim > 2!
        assert int_data.ndim == 2
        assert scale.ndim == 2
        self.int_data = int_data
        self.scale = scale

    # __tensor_flatten__ returns a tuple of:
    # - names of all inner tensor attributes (two in our case)
    # - any other additional, non-tensor metadata.
    def __tensor_flatten__(self) -> Tuple[List[str], Any]:
        return ["int_data", "scale"], None

    # __tensor_unflatten__ should effectively undo __tensor_flatten__.
    # inputs:
    # - a dict mapping names of inner tensor attributes back to the tensors
    # - the constant metadata from __tensor_flatten__
    # output:
    # - a new instance of your subclass
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, extra_metadata, outer_size=None, outer_stride=None
    ):
        assert extra_metadata is None
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        return Int8SymmetricTensor(int_data, scale)

    def __repr__(self):
        return f"Int8SymmetricTensor(int_data={repr(self.int_data)}, scale={repr(self.scale)})"

    # Actually performs the symmetric quantization.
    # In our simple inference example we will quantize weights "ahead-of-time",
    # although later in a training example we can quantize/dequantize
    # during model execution, inside of our __torch_dispatch__
    # input:
    # - float32 torch.Tensor
    # output:
    # - Int8SymmetricTensor
    @staticmethod
    def from_float(float_tensor):
        int8_tensor, scale = int8_symmetric_quantize(float_tensor)
        return Int8SymmetricTensor(int8_tensor, scale)

    # __torch_dispatch__ gets called for ATen operator
    # that our subclass is passed as an input to.
    # We need to define our own implementation for every operator here.
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(
                f"Int8SymmetricTensor does not yet support op: {str(func)}"
            )
        return op_implementations_dict[func](func, *args, **kwargs)


# Convenience function for registering our own implementation
# to every ATen operator in PyTorch
op_implementations_dict = {}


def register_op(ops: List[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict
        for op in ops:
            op_implementations_dict[op] = op_impl
        return op_impl

    return impl_decorator


from torch.utils._python_dispatch import return_and_correct_aliasing


# matmul impl
@register_op([torch.ops.aten.mm.default])
def int8_mm(func, x, weight):
    assert isinstance(
        weight, Int8SymmetricTensor
    ), "Int8SymmetricTensor: matmul currently only supports the weight in low precision, not the input!"
    return torch.mm(x, weight.int_data.to(x.dtype)) * weight.scale


# implementation of most view operations
@register_op(
    [
        torch.ops.aten.detach.default,
        torch.ops.aten.t.default,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
    ]
)
def int8_view_ops(func, *args, **kwargs):
    assert isinstance(args[0], Int8SymmetricTensor)
    out_data = func(args[0].int_data, *args[1:], **kwargs)
    out_scale = func(args[0].scale, *args[1:], **kwargs)
    out = Int8SymmetricTensor(out_data, out_scale)
    # "return_and_correct_aliasing" here is needed for torch.compile support.
    # It effectively tells the compiler that the output of this view op aliases its input.
    # At some point, we're hoping to infer this automatically and kill this extra API!
    return return_and_correct_aliasing(func, args, kwargs, out)


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
    quantized_model_subclass = copy.deepcopy(float_model)

    # Swap torch.nn.Linear weights with Int8SymmetricTensor subclasses
    for name, child in quantized_model_subclass.named_children():
        if type(child) == torch.nn.Linear:
            subclass_param = Int8SymmetricTensor.from_float(child.weight)
            child.weight = torch.nn.Parameter(subclass_param, requires_grad=True)

    with torch.no_grad():
        x = torch.randn(64, 64, 64, device="cuda")
        out = quantized_model_subclass(x)

        # We can also use torch.compile to fuse some of our quantized logic
        # run with TORCH_LOGS="output_code" to see the generated inductor code
        out_compiled = torch.compile(quantized_model_subclass)(x)
        print(torch.allclose(out, out_compiled))
