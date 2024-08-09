import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.utils import _dispatch__torch_dispatch__, _dispatch__torch_function__, _implements
from torchao.quantization.quant_api import _get_linear_subclass_inserter

aten = torch.ops.aten


# the main difference of this tensor subclass from AffineQuantizedTensor:
# 1. F.linear is differentiable i.e. backward is defined.
# 2. support stochastic rounding when casting from floating point.
class Int8QTLinearWeight(Tensor):
    implements = classmethod(_implements)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)

    def __new__(cls, int_data, scale, requires_grad=False):
        return Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            dtype=scale.dtype,
            device=int_data.device,
            requires_grad=requires_grad,
        )

    def __init__(self, int_data, scale, requires_grad=False):
        """Create a symmetric quantized INT8 weight. This tensor will appear to have the same dtype
        as `scale.dtype`. All in-place update ops will perform stochastic rounding.
        """
        # NOTE: should scale always be FP32?
        assert int_data.dtype is torch.int8
        self.int_data = int_data
        self.scale = scale

    def __tensor_flatten__(self):
        return ["int_data", "scale"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes)

    @staticmethod
    def quantize(tensor: Tensor, stochastic_rounding: bool = False):
        original_dtype = tensor.dtype
        tensor = tensor.float()

        # absmax symmetric quantization
        scale = tensor.abs().amax(-1) / 127
        tensor = tensor / scale.clip(1e-12).view(-1, 1)

        if stochastic_rounding:
            # floor is required since .to(torch.int8) will convert 3.1 to 3 but -3.1 to -3
            tensor = (tensor + torch.rand_like(tensor)).floor()
        else:
            tensor = tensor.round()

        # NOTE: is clipping necessary?
        tensor = tensor.clip(-128, 127).to(torch.int8)
        return tensor, scale.to(original_dtype)

    @classmethod
    def from_float(cls, tensor: Tensor):
        """Convert a float tensor into INT8 quantized weight. No stochastic rounding is performed."""
        int_data, scale = cls.quantize(tensor.detach())
        return cls(int_data, scale, requires_grad=tensor.requires_grad)

    def dequantize(self):
        return self.int_data * self.scale.view(-1, 1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )


@Int8QTLinearWeight.implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    return _Int8WeightOnlyLinear.apply(*args, **kwargs)


@Int8QTLinearWeight.implements(aten.detach.default)
def _(func, types, args, kwargs):
    out = Int8QTLinearWeight(args[0].int_data, args[0].scale, requires_grad=False)
    return return_and_correct_aliasing(func, args, kwargs, out)


@Int8QTLinearWeight.implements(aten.clone.default)
def _(func, types, args, kwargs):
    out = Int8QTLinearWeight(
        args[0].int_data.clone(),
        args[0].scale.clone(),
        requires_grad=args[0].requires_grad,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@Int8QTLinearWeight.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # we ignore memory_format in kwargs
    # only perform dtype casting on scale, which determines the appearance dtype
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    out = Int8QTLinearWeight(
        args[0].int_data.to(device=device),
        args[0].scale.to(device=device, dtype=dtype),
        requires_grad=args[0].requires_grad,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


# to make training work with existing PyTorch optimizers, we return a normal tensor
@Int8QTLinearWeight.implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)


@Int8QTLinearWeight.implements([aten.sub.Tensor, aten.mul.Tensor])
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, Int8QTLinearWeight) else x for x in args]
    return func(*args, **kwargs)


@Int8QTLinearWeight.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    if isinstance(args[0], Int8QTLinearWeight) and isinstance(args[1], Int8QTLinearWeight):
        args[0].int_data.copy_(args[1].int_data)
        args[0].scale.copy_(args[1].scale)

    elif isinstance(args[0], Int8QTLinearWeight):
        int_data, scale = Int8QTLinearWeight.quantize(args[1], stochastic_rounding=True)
        args[0].int_data.copy_(int_data)
        args[0].scale.copy_(scale)

    else:
        args[0].copy_(args[1].dequantize())

    return args[0]


@Int8QTLinearWeight.implements(aten.addcdiv_.default)
def _(func, types, args, kwargs):
    out = torch.addcdiv(args[0].dequantize(), *args[1:], **kwargs)
    return args[0].copy_(out)


@Int8QTLinearWeight.implements(aten.add_.Tensor)
def _(func, types, args, kwargs):
    out = torch.add(args[0].dequantize(), *args[1:], **kwargs)
    return args[0].copy_(out)


class _Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8QTLinearWeight, bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
        out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        dinput = (grad_output * weight.scale) @ weight.int_data.to(grad_output.dtype)
        dweight = grad_output.flatten(0, -2).T @ input.flatten(0, -2)
        dbias = grad_output.sum(0) if ctx.bias else None
        return dinput, dweight, dbias


def int8_weight_only_quantized_training():
    def apply_int8_linear_weight(linear: nn.Linear):
        linear.weight = nn.Parameter(
            Int8QTLinearWeight.from_float(linear.weight),
            requires_grad=linear.weight.requires_grad,
        )
        return linear

    return apply_int8_linear_weight
