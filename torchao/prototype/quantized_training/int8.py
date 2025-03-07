from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional


@torch.no_grad()
def quantize_int8_rowwise(
    tensor: Tensor, stochastic_rounding: bool = False, eps: float = 1e-12
):
    """Normal rounding will always round down small changes in weight update. To tackle this problem,
    stochastic rounding can be used, which has a low chance, but not zero, of rounding up. The
    probability of rounding up is equal to x - ⌊x⌋, which indicates how close the value is to the next
    integer value. Thus, stochastic rounding also approximates the floating point value exactly.

    Currently this function differs from AQT's `int8_weight_only()` in the following way:
    1. Precision: AQT keeps original dtype when doing quantization, while this function upcasts input
    to FP32 before quantization. Output scale maintains the original input dtype.
    2. Calculate scale: AQT uses `input.abs().amax() / 127.5`, while `input.abs().amax() / 127` is
    done here.
    3. Apply scale: AQT uses `input * (1 / scale)`, while this function performs `input / scale`.
    """
    # absmax symmetric quantization
    scale = tensor.abs().amax(1) / 127  # same dtype as tensor
    inv_scale = 1.0 / scale.float().clip(eps)
    tensor = tensor.float() * inv_scale.view(
        -1, 1
    )  # slightly faster than divide directly

    if stochastic_rounding:
        tensor = (tensor + torch.rand_like(tensor)).floor()
    else:
        tensor = tensor.round()

    tensor = tensor.clip(-128, 127).to(torch.int8)
    return tensor, scale


class Int8QuantizedTrainingLinearWeight(TorchAOBaseTensor):
    """INT8 symmetric quantization weight, with absmax scaling [-127, 127]. The main difference
    of this tensor subclass from AffineQuantizedTensor:
    1. `F.linear` is differentiable i.e. backward is defined.
    2. All in-place ops, such as `aten.copy_`, will perform stochastic rounding.
        `Int8QTLinearWeight.from_float()` does not perform stochastic rounding.
    3. The numerics for quantization is slightly different. See `quantize_int8_rowwise()`
        for more details.
    """

    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: Tensor, scale: Tensor):
        return Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: Tensor, scale: Tensor):
        """Create a symmetric quantized INT8 weight. This tensor will appear to have the same dtype
        as `scale.dtype`. All in-place update ops will perform stochastic rounding.
        """
        # NOTE: should scale always be FP32?
        assert int_data.dtype is torch.int8
        assert int_data.ndim == 2
        assert scale.ndim == 1
        self.int_data = int_data
        self.scale = scale

    def __tensor_flatten__(self):
        return ["int_data", "scale"], []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes
        )

    @classmethod
    def from_float(cls, tensor: Tensor):
        """Convert a float tensor into INT8 quantized weight. No stochastic rounding is performed.
        This function is not differentiable.
        """
        int_data, scale = quantize_int8_rowwise(tensor.detach())
        out = cls(int_data, scale)
        out.requires_grad_(tensor.requires_grad)
        return out

    def dequantize(self):
        return self.int_data * self.scale.view(-1, 1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )

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
        scale = self.scale
        if mp_policy is not None:
            scale = scale.to(mp_policy.param_dtype)

        return (self.int_data, scale), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Tensor] = None,
    ):
        int_data, scale = all_gather_outputs
        return Int8QuantizedTrainingLinearWeight(int_data, scale), all_gather_outputs


class _Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Int8QuantizedTrainingLinearWeight,
        bias: Optional[Tensor] = None,
    ):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
        out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = (grad_output * weight.scale) @ weight.int_data.to(
            grad_output.dtype
        )
        grad_weight = grad_output.view(-1, weight.shape[0]).T @ input.view(
            -1, weight.shape[1]
        )
        grad_bias = grad_output.view(-1, weight.shape[0]).sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias


implements = Int8QuantizedTrainingLinearWeight.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _Int8WeightOnlyLinear.apply(*args, **kwargs)


@implements(
    [
        aten.detach.default,
        aten.clone.default,
        # FSDP ops
        aten.slice.Tensor,
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
        c10d_functional.wait_tensor.default,
        _c10d_functional.wait_tensor.default,
    ]
)
def _(func, types, args, kwargs):
    # will error out if try to slice 2nd dim
    out = Int8QuantizedTrainingLinearWeight(
        func(args[0].int_data, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # only perform dtype casting on scale, which determines the appearance dtype
    # TODO: handle non_blocking kwarg?
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    out = Int8QuantizedTrainingLinearWeight(
        args[0].int_data.to(device=device),
        args[0].scale.to(device=device, dtype=dtype),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


# to make training work with existing PyTorch optimizers, we return a normal tensor
@implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)


# out-of-place math ops always return plain tensor
@implements([aten.sub.Tensor, aten.mul.Tensor])
def _(func, types, args, kwargs):
    args = [
        x.dequantize() if isinstance(x, Int8QuantizedTrainingLinearWeight) else x
        for x in args
    ]
    return func(*args, **kwargs)


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    if isinstance(args[0], Int8QuantizedTrainingLinearWeight) and isinstance(
        args[1], Int8QuantizedTrainingLinearWeight
    ):
        args[0].int_data.copy_(args[1].int_data, **kwargs)
        args[0].scale.copy_(args[1].scale, **kwargs)

    elif isinstance(args[0], Int8QuantizedTrainingLinearWeight):
        int_data, scale = quantize_int8_rowwise(args[1], stochastic_rounding=True)
        args[0].int_data.copy_(int_data, **kwargs)
        args[0].scale.copy_(scale, **kwargs)

    else:
        args[0].copy_(args[1].dequantize(), **kwargs)

    return args[0]


@implements([aten.addcdiv_.default, aten.add_.Tensor])
def _(func, types, args, kwargs):
    original = args[0]
    out = func(args[0].dequantize(), *args[1:], **kwargs)
    return original.copy_(out)


# FSDP ops
@implements(aten.split.Tensor)
def _(func, types, args, kwargs):
    if len(args) == 3 and args[2] != 0:
        raise NotImplementedError("Int8QTLinearWeight only supports split at dim=0")

    int8_weight: Int8QuantizedTrainingLinearWeight = args[0]
    int_data_list = func(int8_weight.int_data, *args[1:], **kwargs)
    scale_list = func(int8_weight.scale, *args[1:], **kwargs)

    out = [
        Int8QuantizedTrainingLinearWeight(int_data, scale)
        for int_data, scale in zip(int_data_list, scale_list)
    ]
    return out


@implements(aten.new_zeros.default)
def _(func, types, args, kwargs):
    size = args[1]
    if len(size) != 2:
        raise NotImplementedError

    # TODO: handle pin_memory kwarg?
    device = kwargs.get("device", args[0].device)
    dtype = kwargs.get("dtype", args[0].dtype)
    int_data = torch.zeros(size, device=device, dtype=torch.int8)
    scale = torch.zeros(size[0], device=device, dtype=dtype)
    return Int8QuantizedTrainingLinearWeight(int_data, scale)


# FSDP2 will call these two ops, expecting a view, not a copy. It doesn't make sense to
# correctly support these ops. For example, `.scale` depends on the shape of the weight,
# since this is channel-wise quantization.
# Thus, this is a workaround for FSDP2. Users SHOULD NOT call these ops directly, since
# they will produce unexpected or wrong results.
@implements([aten.view.default, aten.as_strided.default])
def _(func, types, args, kwargs):
    out = Int8QuantizedTrainingLinearWeight(args[0].int_data, args[0].scale)
    return return_and_correct_aliasing(func, args, kwargs, out)


class Int8WeightOnlyQuantizedTrainingConfig(AOBaseConfig):
    pass


# for bc
int8_weight_only_quantized_training = Int8WeightOnlyQuantizedTrainingConfig


@register_quantize_module_handler(Int8WeightOnlyQuantizedTrainingConfig)
def _int8_weight_only_quantized_training_transform(
    module: torch.nn.Module,
    config: Int8WeightOnlyQuantizedTrainingConfig,
) -> torch.nn.Module:
    new_weight = Int8QuantizedTrainingLinearWeight.from_float(module.weight)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=True)
    return module
