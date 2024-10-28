from typing import Any, NamedTuple, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._triton import has_triton

from torchao.quantization.quant_api import _get_linear_subclass_inserter
from torchao.utils import TorchAOBaseTensor

from .int8 import quantize_int8_rowwise

if has_triton():
    from .int8_mm import scaled_int8_mm

else:

    # This is less performant than the explicit hand-written Triton kernel, though things might
    # change in the future.
    # Multiplying col_scale first is faster than the other way round.
    def scaled_int8_mm(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
        return torch._int_mm(A, B) * col_scale.view(-1) * row_scale.view(-1, 1)


class Int8MixedPrecisionTrainingConfig(NamedTuple):
    output: bool = True
    grad_input: bool = True
    grad_weight: bool = True


_DEFAULT_CONFIG = Int8MixedPrecisionTrainingConfig()


aten = torch.ops.aten


class Int8MixedPrecisionTrainingLinearWeight(TorchAOBaseTensor):
    """Linear weight for INT8 mixed-precision training. The weight is in original precision (e.g. FP32 or BF16).
    During training, weight and activation are dynamically quantized and cast to INT8 to utilize INT8 Tensor Cores,
    and then scaled back to original precision. This is also applied to backward pass.
    """

    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, config: Int8MixedPrecisionTrainingConfig):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            data.stride(),
            data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, config: Int8MixedPrecisionTrainingConfig):
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

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
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

        out = func(
            *pytree.tree_map_only(cls, unwrap, args),
            **pytree.tree_map_only(cls, unwrap, kwargs),
        )

        if func is aten.copy_.default:
            # return original object
            return args[0]
        elif func in {
            aten.t.default,
            aten.detach.default,
            aten.empty_like.default,
            aten.new_zeros.default,
            aten.slice.Tensor,
            aten.view.default,
            aten.as_strided.default,
            aten._to_copy.default,
            aten._pin_memory.default,
            aten.split.Tensor,
            aten.clone.default,
        }:
            # return new wrapped object
            return pytree.tree_map_only(Tensor, lambda x: cls(x, config), out)
        else:
            # return new unwrapped object
            return out

    # FSDP all-gather extension v2
    # https://github.com/pytorch/pytorch/pull/137005
    # we need default values so this method still works with PyTorch 2.4 and 2.5
    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size=None,
        outer_stride=None,
        module=None,
        mp_policy=None,
    ):
        # TODO: pre-quantize weight here -> reduce comm bandwidth.
        # we will need another tensor subclass to hold the quantized weight.
        data = self._data
        if mp_policy is not None:
            data = data.to(mp_policy.param_dtype)

        return (data,), (self.config,)

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
        if out is not None:
            assert isinstance(out, Int8MixedPrecisionTrainingLinearWeight)
            assert out.config == config
            return
        return Int8MixedPrecisionTrainingLinearWeight(data, config), all_gather_outputs


@Int8MixedPrecisionTrainingLinearWeight.implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_gpu_dtype()
        args = tuple(x.to(dtype) if x is not None else x for x in args)
    return _Int8MixedPrecisionTrainingLinear.apply(*args, **kwargs)


def _dynamic_int8_mm(A: Tensor, B: Tensor) -> Tensor:
    """Dynamically quantize A and B to perform INT8 matmul, then scale the results back to original precision.
    To fuse scaling to matmul output, we use row-wise scaling for A and column-wise scaling for B.

    We transpose B before quantization for 2 reasons:
      - INT8 matmul is the most performant when A is row-major and B is column-major.
      - Row-wise scaling for B.T is column-wise scaling for B -> we only need to implement row-wise scaling.

    Note that inputs and outputs of `quantize_int8_rowwise()` are not guaranteed to be contiguous. We call
    `.contiguous()` to outputs of the quantize op to make sure:
      - Performant layout for INT8 matmul inputs (see above).
      - Scales are contiguous (this is a limitation of our triton kernel).

    We hope that the `.contiguous()` calls, as well as possible layout transpose before quantization, are
    fused into quantize op by torch compiler.

    TODO: check if transpose+quantize are actually fused.
    """
    # A may have more than 2 dims, while B must be exactly 2-dim
    A_i8, A_scale_rowwise = quantize_int8_rowwise(A.view(-1, A.shape[-1]))
    B_t_i8, B_scale_colwise = quantize_int8_rowwise(B.T)
    out = scaled_int8_mm(
        A_i8.contiguous(),
        B_t_i8.contiguous().T,
        A_scale_rowwise.contiguous(),
        B_scale_colwise.contiguous(),
    )
    return out.view(*A.shape[:-1], out.shape[-1])


class _Int8MixedPrecisionTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(input: Tensor, weight: Int8MixedPrecisionTrainingLinearWeight, bias: Optional[Tensor]):
        if weight.config.output:
            out = _dynamic_int8_mm(input, weight._data.T)
        else:
            out = input @ weight._data.T
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.config = weight.config
        ctx.save_for_backward(input, weight._data)
        ctx.bias = bias is not None

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            if ctx.config.grad_input:
                grad_input = _dynamic_int8_mm(grad_output, weight)
            else:
                grad_input = grad_output @ weight

        if ctx.needs_input_grad[1]:
            grad_output = grad_output.view(-1, weight.shape[0])
            input = input.view(-1, weight.shape[1])
            if ctx.config.grad_weight:
                # grad_weight = _dynamic_int8_mm(grad_output.T, input)
                grad_weight = _dynamic_int8_mm(input.T, grad_output).T  # this is slightly faster
            else:
                grad_weight = grad_output.T @ input

        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def int8_mixed_precision_training(config: Int8MixedPrecisionTrainingConfig = _DEFAULT_CONFIG):
    return _get_linear_subclass_inserter(
        Int8MixedPrecisionTrainingLinearWeight,
        config=config,
        allow_requires_grad=True,
    )
