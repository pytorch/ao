from typing import Any, NamedTuple, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import Tensor, nn
from torch.utils._triton import has_triton

from .int8 import quantize_int8_rowwise

if has_triton():
    from .int8_mm import int8_mm_dequant

else:

    def int8_mm_dequant(A: Tensor, B: Tensor, A_scale_rowwise: Tensor, B_scale_colwise: Tensor) -> Tensor:
        A_scaled = A * A_scale_rowwise.view(-1, 1)
        B_scaled = B * B_scale_colwise.view(1, -1)
        return A_scaled @ B_scaled


class Int8MixedPrecisionConfig(NamedTuple):
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = True


_DEFAULT_CONFIG = Int8MixedPrecisionConfig()


# adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
aten = torch.ops.aten
_ops_to_preserve_subclass = {
    aten.detach.default,
    aten.empty_like.default,
    aten.new_zeros.default,
    aten.slice.Tensor,
    aten.copy_.default,
    aten.view.default,
    aten.as_strided.default,
    aten._to_copy.default,
    aten._pin_memory.default,
    aten.split.Tensor,
    aten.clone.default,
}


class Int8MixedPrecisionLinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, config: Int8MixedPrecisionConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            pin_memory=data.is_pinned(),
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
        return f"{self.__class__.__name__}(data={self._data}, config={self.config})"

    def to_original(self):
        return self._data.clone()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = dict()
        
        if func is torch.nn.functional.linear:
            act, weight = args[:2]
            bias = args[2] if len(args) > 2 else None
            return _Int8MixedPrecisionLinear.apply(act, weight._data, bias, weight.config)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        config = None

        def unwrap(x: cls):
            nonlocal config
            if config is None:
                config = x.config
            else:
                assert x.config == config
            return x._data

        args, kwargs = pytree.tree_map_only(cls, unwrap, (args, kwargs))
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        else:
            return pytree.tree_map_only(Tensor, lambda x: cls(x, config), out)

    def fsdp_pre_all_gather(self, mesh):
        # TODO: pre-quantize weight here -> reduce comm bandwidth.
        # we will need another tensor subclass to hold the quantized weight.
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
        return Int8MixedPrecisionLinearWeight(data.to(param_dtype), config), all_gather_outputs


# alternative UX
class Int8MixedPrecisionLinear(nn.Linear):
    def __init__(self, *args, config: Int8MixedPrecisionConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    def forward(self, input: Tensor) -> Tensor:
        return _Int8MixedPrecisionLinear.apply(input, self.weight, self.bias, self.config)

    def extra_repr(self):
        return f"{super().extra_repr()}, config={self.config}"

    @classmethod
    def convert_linear(cls, module: nn.Module, config: Int8MixedPrecisionConfig = _DEFAULT_CONFIG):
        if module.__class__ is nn.Linear:  # exact match, don't swap nn.Linear subclasses
            module.__class__ = cls
            module.config = config
            return
        for child in module.children():
            cls.convert_linear(child, config)


def _dynamic_int8_linear(input: Tensor, weight: Tensor) -> Tensor:
    # TODO: check if we need to enforce .contiguous() for input_i8 and weight_i8
    input_i8, input_scale = quantize_int8_rowwise(input)
    weight_i8, weight_scale = quantize_int8_rowwise(weight)
    return int8_mm_dequant(input_i8, weight_i8.T, input_scale, weight_scale)


class _Int8MixedPrecisionLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor], config: Int8MixedPrecisionConfig):
        ctx.config = config
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        if ctx.config.output:
            batch_dims = input.shape[:-1]
            input = input.view(-1, weight.shape[1])
            out = _dynamic_int8_linear(input, weight)
            out = out.view(*batch_dims, weight.shape[0])
        else:
            out = input @ weight.T

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_config = None

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight.shape[0])
        input = input.view(-1, weight.shape[1])

        if ctx.needs_input_grad[0]:
            if ctx.config.grad_input:
                grad_input = _dynamic_int8_linear(grad_output, weight.T)
            else:
                grad_input = grad_output @ weight
            grad_input = grad_input.view(*batch_dims, weight.shape[1])

        if ctx.needs_input_grad[1]:
            if ctx.config.grad_weight:
                # TODO: check if transpose+quantize are fused
                # grad_weight = _dynamic_int8_linear(grad_output.T, input.T)
                grad_weight = _dynamic_int8_linear(input.T, grad_output.T).T  # this is slightly faster
            else:
                grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, grad_config


def int8_mixed_precision_training(config: Int8MixedPrecisionConfig = _DEFAULT_CONFIG):
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
