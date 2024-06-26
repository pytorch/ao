import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    MappingType,
    ZeroPointDomain,
)


# re-use AffineQuantizedTensor?
class DynamicInt8(Tensor):
    implements = classmethod(_implements)
    tensor_attrs = ["int_data", "scale", "zero_point"]

    @staticmethod
    def __new__(cls, int_data: Tensor, scale: Tensor, zero_point: Tensor):
        return Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            device=int_data.device,
            requires_grad=False,
        )

    def __init__(self, int_data: Tensor, scale: Tensor, zero_point: Tensor):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point

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
        pass

    def dequantize(self, output_dtype=None):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"device={self.device}, requires_grad={self.requires_grad})"
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(*[fn(getattr(self, name)) for name in self.tensor_attrs])

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

        raise NotImplementedError(f"{cls.name} dispatch: attempting to run {func}, this is not supported")
