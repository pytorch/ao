import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    MappingType,
    ZeroPointDomain,
)


aten = torch.ops.aten


# re-use AffineQuantizedTensor?
class DynamicInt8(Tensor):
    implements = classmethod(_implements)
    tensor_attrs = ["int_data", "scale", "zero_point"]

    @staticmethod
    def __new__(cls, int_data: Tensor, scale: Tensor, zero_point: Tensor, shape):
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=int_data.device,
            requires_grad=False,
        )

    def __init__(self, int_data: Tensor, scale: Tensor, zero_point: Tensor, shape):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point

    @property
    def group_size(self):
        return self.numel() // self.scale.shape[0]

    def __tensor_flatten__(self):
        return self.tensor_attrs, []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs],
            *tensor_attributes,
        )

    @classmethod
    def from_float(cls, input_float: Tensor, group_size: int):
        shape = input_float.shape
        input_float = input_float.flatten()

        scale, zero_point = choose_qparams_affine(
            input_float,
            MappingType.ASYMMETRIC,
            (group_size,),
            torch.uint8,
            preserve_zero=False,
            zero_point_domain=ZeroPointDomain.FLOAT,
        )
        int_data = quantize_affine(
            input_float,
            (group_size,),
            scale,
            zero_point,
            torch.uint8,
            zero_point_domain=ZeroPointDomain.FLOAT,
        )
        return cls(int_data, scale, zero_point, shape)

    def dequantize(self, output_dtype=None):
        return dequantize_affine(
            self.int_data,
            (self.group_size,),
            self.scale,
            self.zero_point,
            torch.uint8,
            zero_point_domain=ZeroPointDomain.FLOAT,
        ).view(self.shape)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(group_size={self.group_size}, shape={tuple(self.shape)}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(*[fn(getattr(self, name)) for name in self.tensor_attrs], self.shape)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


def _dequant_list(*args):
    return [x.dequantize() if isinstance(x, DynamicInt8) else x for x in args]


# in-place ops
@DynamicInt8.implements([aten.add_.Tensor, aten.mul_.Tensor, aten.addcmul_.default, aten.addcdiv_.default, aten.lerp_.Scalar])
def _(func, *args, **kwargs):
    out = func(*_dequant_list(*args), **kwargs)

    # args[0] is the original quantized tensor to be updated in-place
    if isinstance(args[0], DynamicInt8):
        out = DynamicInt8.from_float(out, args[0].group_size)
        args[0].int_data.copy_(out.int_data)
        args[0].scale.copy_(out.scale)
        args[0].zero_point.copy_(out.zero_point)

        # return the original quantized tensor with updated values
        out = args[0]

    return out


# out-of-place ops will always return float tensor
@DynamicInt8.implements([aten.sqrt.default, aten.div.Tensor])
def _(func, *args, **kwargs):
    return func(*_dequant_list(*args), **kwargs)
