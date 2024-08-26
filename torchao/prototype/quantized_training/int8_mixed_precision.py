from typing import Any, NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._triton import has_triton

from torchao.dtypes.utils import _dispatch__torch_dispatch__, _dispatch__torch_function__, _implements

from .int8 import quantize_int8_rowwise

if has_triton():
    from .int8_mm import int8_mm_dequant

else:

    def int8_mm_dequant(A: Tensor, B: Tensor, A_scale_rowwise: Tensor, B_scale_colwise: Tensor) -> Tensor:
        return (A * A_scale_rowwise.view(-1, 1)) @ (B * B_scale_colwise.view(1, -1))


aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional


class Int8MixedPrecisionConfig(NamedTuple):
    forward: bool = False
    backward_grad_input: bool = False
    backward_grad_weight: bool = False


class Int8MixedPrecisionLinearWeight(Tensor):
    implements = classmethod(_implements)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)

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

    def to_original(self):
        return self._data.clone()

    def fsdp_pre_all_gather(self, mesh):
        return (self._data,), (self.config,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (config,) = metadata
        return Int8MixedPrecisionLinearWeight(data, config), all_gather_outputs


implements = Int8MixedPrecisionLinearWeight.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _Int8MixedPrecisionLinear.apply(*args, **kwargs)


@implements(
    [
        aten.detach.default,
        aten.clone.default,
        aten._to_copy.default,
        # FSDP ops
        aten.slice.Tensor,
        aten.new_zeros.default,
        aten.view.default,
        aten.as_strided.default,
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
        c10d_functional.wait_tensor.default,
        _c10d_functional.wait_tensor.default,
    ]
)
def _(func, types, args, kwargs):
    out = Int8MixedPrecisionLinearWeight(func(args[0]._data, *args[1:], **kwargs), args[0].config)
    return return_and_correct_aliasing(func, args, kwargs, out)


@implements(
    [
        aten.copy_.default,
        aten.addcdiv_.default,
        aten.add_.Tensor,
        aten.mul_.Tensor,
    ]
)
def _(func, types, args, kwargs):
    unpacked_args = [x._data if isinstance(x, Int8MixedPrecisionLinearWeight) else x for x in args]
    func(*unpacked_args, **kwargs)
    return args[0]


# called by optimizers. return a normal tensor
@implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    return func(args[0]._data, *args[1:], **kwargs)


# FSDP op
@implements(aten.split.Tensor)
def _(func, types, args, kwargs):
    data_list = func(args[0]._data, *args[1:], **kwargs)
    return [Int8MixedPrecisionLinearWeight(x, args[0].config) for x in data_list]


class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8MixedPrecisionLinearWeight, bias: Optional[Tensor] = None):
        ctx.config = weight.config
        weight = weight._data
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        if ctx.config.forward:
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

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        if ctx.needs_input_grad[0]:
            if ctx.config.backward_grad_input:
                grad_output_i8, grad_output_scale = quantize_int8_rowwise(grad_output)
                weight_i8_t, weight_scale = quantize_int8_rowwise(weight.T)
                grad_input = int8_mm_dequant(grad_output_i8, weight_i8_t.T, grad_output_scale, weight_scale)
            else:
                grad_input = grad_output @ weight
            grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if ctx.needs_input_grad[1]:
            if ctx.config.backward_grad_weight:
                grad_output_i8_t, grad_output_scale = quantize_int8_rowwise(grad_output.T)
                input_i8_t, input_scale = quantize_int8_rowwise(input.T)
                grad_weight = int8_mm_dequant(grad_output_i8_t, input_i8_t.T, grad_output_scale, input_scale)
            else:
                grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# NOTE: should default config set all to True instead? -> speedup out-of-the-box.
# only if there are convergence issues, turn off some INT8 matmuls in backward.
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
