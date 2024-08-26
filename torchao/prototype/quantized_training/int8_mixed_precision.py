from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils._triton import has_triton

from .int8 import quantize_int8_rowwise

if has_triton():
    from .int8_mm import int8_mm_dequant

else:

    def int8_mm_dequant(A: Tensor, B: Tensor, A_scale_rowwise: Tensor, B_scale_colwise: Tensor) -> Tensor:
        return (A * A_scale_rowwise.view(-1, 1)) @ (B * B_scale_colwise.view(1, -1))


aten = torch.ops.aten


class Int8MixedPrecisionConfig(NamedTuple):
    forward: bool = False
    backward_grad_input: bool = False
    backward_grad_weight: bool = False


class Int8MixedPrecisionLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, config: Int8MixedPrecisionConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, config: Int8MixedPrecisionConfig):
        self._data = data
        self.config = config

    def __tensor_flatten__(self):
        return ["_data"], [self.config]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"], *tensor_attributes)

    def __repr__(self):
        return self._data.__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is torch.nn.functional.linear:
            return _Int8MixedPrecisionLinear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in (aten.detach.default, aten.clone.default, aten._to_copy.default):
            return cls(func(args[0]._data, *args[1:], **kwargs), args[0].config)

        # TODO: some ops should return the original class i.e. in-place ops
        args = [x._data if isinstance(x, cls) else x for x in args]
        return func(*args, **kwargs)


class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8MixedPrecisionLinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        if weight.config.forward:
            batch_dims = input.shape[:-1]
            input = input.view(-1, weight.shape[1])
            input_i8, input_scale = quantize_int8_rowwise(input)
            weight_i8, weight_scale = quantize_int8_rowwise(weight)
            out = int8_mm_dequant(input_i8, weight_i8.T, input_scale, weight_scale)
            out = out.view(*batch_dims, weight.shape[0])
        else:
            out = input @ weight.T

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight: Int8MixedPrecisionLinearWeight

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        if ctx.needs_input_grad[0]:
            if weight.config.backward_grad_input:
                grad_output_i8, grad_output_scale = quantize_int8_rowwise(grad_output)
                weight_i8_t, weight_scale = quantize_int8_rowwise(weight.T)
                grad_input = int8_mm_dequant(grad_output_i8, weight_i8_t.T, grad_output_scale, weight_scale)
            else:
                grad_input = grad_output @ weight
            grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if ctx.needs_input_grad[1]:
            if weight.config.backward_grad_weight:
                grad_output_i8_t, grad_output_scale = quantize_int8_rowwise(grad_output.T)
                input_i8_t, input_scale = quantize_int8_rowwise(input.T)
                grad_weight = int8_mm_dequant(grad_output_i8_t, input_i8_t.T, grad_output_scale, input_scale)
            else:
                grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def int8_mixed_precision_training(config: Int8MixedPrecisionConfig = Int8MixedPrecisionConfig()):
    # TODO: right now `_get_linear_subclass_inserter()` will always set `requires_grad=False`
    # when we have this out of prototype (or there are stable trainable tensor subclasses),
    # update `_get_linear_subclass_inserter()` to allow `requires_grad=True`.
    def apply_int8_linear_weight(linear: nn.Linear):
        linear.weight = nn.Parameter(
            Int8MixedPrecisionLinearWeight(linear.weight.detach(), config),
            requires_grad=linear.weight.requires_grad,
        )
        return linear

    return apply_int8_linear_weight
