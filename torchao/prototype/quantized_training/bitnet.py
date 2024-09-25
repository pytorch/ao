# this file implements BitNet b1.58 https://arxiv.org/abs/2402.17764
# a reference implementation is available at
# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
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
        return torch._int_mm(A, B) * col_scale * row_scale.view(-1, 1)


aten = torch.ops.aten


class BitNetTrainingLinearWeight(TorchAOBaseTensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor):
        self._data = data

    def __tensor_flatten__(self):
        return ["_data"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"], *tensor_attributes)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        out = func(
            *pytree.tree_map_only(cls, lambda x: x._data, args),
            **pytree.tree_map_only(cls, lambda x: x._data, kwargs),
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
            return pytree.tree_map_only(Tensor, lambda x: cls(x), out)
        else:
            # return new unwrapped object
            return out

    def fsdp_pre_all_gather(self, mesh):
        # quantize and pack into 2-bit to save comm bandwidth
        # TODO: precompute absmean similar to float8
        data = BitNetPacked2bitLinearWeight.from_float(self._data, all_reduce=True)
        return (data.int_data,), (data.scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        (int_data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, BitNetPacked2bitLinearWeight)
            out.scale = scale
            return
        return BitNetPacked2bitLinearWeight(int_data, scale), all_gather_outputs


@BitNetTrainingLinearWeight.implements(F.linear)
def _(func, types, args, kwargs):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_gpu_dtype()
        args = tuple(x.to(dtype) if x is not None else x for x in args)
    return _BitNetTrainingLinear.apply(*args, **kwargs)


def quantize_bitnet_weight(w: Tensor, eps: float = 1e-5, all_reduce: bool = False) -> Tensor:
    dtype = w.dtype
    w = w.float()
    scale = w.abs().mean()  # tensor-wise abs-mean. FP32

    if all_reduce and dist.is_initialized():
        dist.all_reduce(scale, op=dist.ReduceOp.AVG)

    w = w / scale.clip(eps)
    w = w.round().clip(-1, 1).to(torch.int8)
    return w, scale.to(dtype)


class _BitNetTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: BitNetTrainingLinearWeight, bias: Optional[Tensor] = None):
        batch_dims = input.shape[:-1]
        input = input.view(-1, weight.shape[1])

        # https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
        # Figure 3
        input_i8, row_scale = quantize_int8_rowwise(input, eps=1e-5)
        weight_i8, tensor_scale = quantize_bitnet_weight(weight._data)

        ctx.save_for_backward(input_i8, row_scale, weight_i8, tensor_scale)

        # use int8 tensor cores
        out = scaled_int8_mm(input_i8.contiguous(), weight_i8.contiguous().T, row_scale, tensor_scale)
        out = out.view(*batch_dims, weight.shape[0])

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_i8, row_scale, weight_i8, tensor_scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight_i8.shape[0])

        # NOTE: we can potentially speedup training by also quantizing the backward pass
        # to use INT8 tensor cores
        if ctx.needs_input_grad[0]:
            # mixed mm
            grad_input = (grad_output @ weight_i8.to(grad_output.dtype)) * tensor_scale
            grad_input = grad_input.view(*batch_dims, weight_i8.shape[1])

        if ctx.needs_input_grad[1]:
            # NOTE: we use quantized activation for this calculation
            grad_weight = grad_output.T @ (input_i8 * row_scale.view(-1, 1))

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def bitnet_training():
    return _get_linear_subclass_inserter(BitNetTrainingLinearWeight, allow_requires_grad=True)


def _pack_i2_to_i8(x: Tensor):
    # NOTE: this is signed integer, so we have to mask before bit-shift
    return (x[:, ::4] << 6) | ((x[:, 1::4] & 0b11) << 4) | ((x[:, 2::4] & 0b11) << 2) | (x[:, 3::4] & 0b11)


def _unpack_i8_to_i2(x: Tensor):
    # NOTE: this is signed integer, so left-shift then right-shift will perform sign extension correctly
    # e.g. aa10bbcc -> 10bbcc00 -> 11111110
    return torch.stack([x >> 6, x << 2 >> 6, x << 4 >> 6, x << 6 >> 6], dim=-1).view(x.shape[0], -1)


# currently this class mainly serves as a container for quantized FSDP2 all-gather,
# so only a minimal set of ops are implemented. this can be extended for inference.
class BitNetPacked2bitLinearWeight(TorchAOBaseTensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: Tensor, scale: Tensor):
        M, N = int_data.shape
        shape = (M, N * 4)
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=scale.dtype,
            device=scale.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: Tensor, scale: Tensor):
        assert int_data.dtype is torch.int8
        assert scale.shape == ()
        self.int_data = int_data
        self.scale = scale

    def __tensor_flatten__(self):
        return ["int_data", "scale"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.dequantize()})"

    @classmethod
    def from_float(cls, tensor: Tensor, *, eps: float = 1e-5, all_reduce: bool = False):
        int_data, scale = quantize_bitnet_weight(tensor, eps=eps, all_reduce=all_reduce)
        int_data = _pack_i2_to_i8(int_data)
        return BitNetPacked2bitLinearWeight(int_data, scale)

    def dequantize(self, out_dtype=None):
        out = _unpack_i8_to_i2(self.int_data) * self.scale
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out


@BitNetPacked2bitLinearWeight.implements(F.linear)
def _(func, types, args, kwargs):
    return _BitNetPacked2bitLinear.apply(*args, **kwargs)


@BitNetPacked2bitLinearWeight.implements(
    [
        aten.detach.default,
        aten.clone.default,
    ]
)
def _(func, types, args, kwargs):
    return BitNetPacked2bitLinearWeight(
        func(args[0].int_data, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
    )


# this is a workaround to make it work with FSDP2.
# end-users should not call this op directly.
@BitNetPacked2bitLinearWeight.implements(aten.as_strided.default)
def _(func, types, args, kwargs):
    return BitNetPacked2bitLinearWeight(args[0].int_data, args[0].scale)


class _BitNetPacked2bitLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: BitNetPacked2bitLinearWeight, bias: Optional[Tensor] = None):
        batch_dims = input.shape[:-1]
        input = input.view(-1, weight.shape[1])

        # https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
        # Figure 3
        input_i8, row_scale = quantize_int8_rowwise(input, eps=1e-5)
        weight_i2, tensor_scale = weight.int_data, weight.scale

        ctx.save_for_backward(input_i8, row_scale, weight_i8, tensor_scale)

        # use int8 tensor cores
        # NOTE: is doing dequant inside matmul faster when M is large?
        weight_i8 = _unpack_i8_to_i2(weight_i2)
        out = scaled_int8_mm(input_i8.contiguous(), weight_i8.contiguous().T, row_scale, tensor_scale)
        out = out.view(*batch_dims, weight.shape[0])

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_i8, row_scale, weight_i2, tensor_scale = ctx.saved_tensors
        weight_i8 = _unpack_i8_to_i2(weight_i2)
        grad_input = grad_weight = grad_bias = None

        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.view(-1, weight_i8.shape[0])

        # NOTE: we can potentially speedup training by also quantizing the backward pass
        # to use INT8 tensor cores
        if ctx.needs_input_grad[0]:
            # mixed mm
            grad_input = (grad_output @ weight_i8.to(grad_output.dtype)) * tensor_scale
            grad_input = grad_input.view(*batch_dims, weight_i8.shape[1])

        if ctx.needs_input_grad[1]:
            # NOTE: we use quantized activation for this calculation
            grad_weight = grad_output.T @ (input_i8 * row_scale.view(-1, 1))

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
