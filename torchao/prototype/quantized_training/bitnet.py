# this file implements BitNet b1.58 https://arxiv.org/abs/2402.17764
# a reference implementation is available at
# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch import Tensor, nn
from torch.utils._triton import has_triton
from torch.distributed._tensor import DTensor

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


aten = torch.ops.aten


class BitNetTrainingLinearWeight(TorchAOBaseTensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, data: Tensor, precomputed_scale: Optional[Tensor] = None):
        return Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            device=data.device,
        )

    @torch._dynamo.disable
    def __init__(self, data: Tensor, precomputed_scale: Optional[Tensor] = None):
        self._data = data
        self._precomputed_scale = precomputed_scale

    def __tensor_flatten__(self):
        if self._precomputed_scale is not None:
            return ["_data", "_precomputed_scale"], []
        else:
            return ["_data"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["_data"], tensor_data_dict.get("_precomputed_scale", None), *tensor_attributes)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self._data})"

    # adapated from FP8 implementation of WeightWithDynamicFloat8CastTensor
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        out = func(
            *pytree.tree_map_only(cls, lambda x: x._data, args),
            **pytree.tree_map_only(cls, lambda x: x._data, kwargs),
        )

        # NOTE: _precomputed_scale does not propagate through any ops
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

    # FSDP all-gather extension v1
    def fsdp_pre_all_gather(self, mesh):
        # quantize and pack into 2-bit to save comm bandwidth
        if self._precomputed_scale is not None:
            scale = self._precomputed_scale

        else:
            scale = get_bitnet_scale(self._data)
            dist.all_reduce(scale, op=dist.ReduceOp.AVG)

        # NOTE: scale is in FP32
        data_i8 = quantize_bitnet_weight(self._data, scale)
        data_i2 = _pack_i2_in_i8(data_i8)
        return (data_i2,), (scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        (data_i2,) = all_gather_outputs
        (scale,) = metadata
        scale = scale.to(param_dtype)
        if out is not None:
            assert isinstance(out, BitNetPacked2bitLinearWeight)
            out.scale = scale
            return
        return BitNetPacked2bitLinearWeight(data_i2, scale), all_gather_outputs


@BitNetTrainingLinearWeight.implements(F.linear)
def _(func, types, args, kwargs):
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_gpu_dtype()
        args = tuple(x.to(dtype) if x is not None else x for x in args)
    return _BitNetTrainingLinear.apply(*args, **kwargs)


def get_bitnet_scale(x: Tensor):
    "Tensor-wise abs-mean. Always return FP32."
    return x.float().abs().mean()


def quantize_bitnet_weight(w: Tensor, scale: Tensor, eps: float = 1e-5) -> Tensor:
    w = w.float() / scale.clip(eps)
    w = w.round().clip(-1, 1).to(torch.int8)
    return w


@torch.no_grad()
def precompute_bitnet_scale_for_fsdp(module: nn.Module):
    """Calculate scale for all BitNetTrainingLinearWeight parameters.
    This should be run after the optimizer step. It performs a single all-reduce for all
    parameters to reduce overhead.
    """
    bitnet_params = [
        p
        for p in module.parameters()
        if isinstance(p, DTensor) and isinstance(p._local_tensor, BitNetTrainingLinearWeight)
    ]
    if len(bitnet_params) == 0:
        return

    # NOTE: use torch.compile to save memory and increase speed?
    bitnet_scales = [get_bitnet_scale(x) for x in bitnet_params]  # local absmean
    bitnet_scales = torch.stack(bitnet_scales)
    bitnet_scales = bitnet_scales.full_tensor()  # global absmean

    for i, p in enumerate(bitnet_params):
        p._local_tensor._precomputed_scale = bitnet_scales[i]


class _BitNetTrainingLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: BitNetTrainingLinearWeight, bias: Optional[Tensor] = None):
        batch_dims = input.shape[:-1]
        input = input.view(-1, weight.shape[1])

        # https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
        # Figure 3
        input_i8, row_scale = quantize_int8_rowwise(input, eps=1e-5)

        # NOTE: use FP32 scale for weight quantization, but cast scale to possibly lower precision
        # for matmul and backward
        tensor_scale = get_bitnet_scale(weight._data)
        weight_i8 = quantize_bitnet_weight(weight._data, tensor_scale)
        tensor_scale = tensor_scale.to(weight.dtype)

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


def _pack_i2_in_i8(x: Tensor):
    # perform packing: [xxxx xxaa, xxxx xxxbb, xxxx xxcc, xxxx xxdd] -> [aabb ccdd]
    # for each value, xxxx can be either all 0s or all 1s because these are signed numbers.
    # thus, we have to mask out the 2 least significant bits (right-most) before bit-shift.
    # e.g. 1111 1111 (value=-1) -> 0000 0011 -> 0011 0000

    x0 = x[:, ::4] << 6  # don't need to mask this number because we shift it to the left-most
    x1 = (x[:, 1::4] & 0b11) << 4
    x2 = (x[:, 2::4] & 0b11) << 2
    x3 = x[:, 3::4] & 0b11
    return x0 | x1 | x2 | x3


def _unpack_i2_in_i8(x: Tensor):
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

    def dequantize(self, out_dtype=None):
        out = _unpack_i2_in_i8(self.int_data) * self.scale
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

        ctx.save_for_backward(input_i8, row_scale, weight_i2, tensor_scale)

        # use int8 tensor cores
        # NOTE: is doing dequant inside matmul faster when M is large?
        weight_i8 = _unpack_i2_in_i8(weight_i2)
        out = scaled_int8_mm(input_i8.contiguous(), weight_i8.contiguous().T, row_scale, tensor_scale)
        out = out.view(*batch_dims, weight.shape[0])

        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_i8, row_scale, weight_i2, tensor_scale = ctx.saved_tensors
        weight_i8 = _unpack_i2_in_i8(weight_i2)
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
