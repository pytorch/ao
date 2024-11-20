from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing, is_traceable_wrapper_subclass

from torchao.dtypes.affine_quantized_tensor import register_layout, AffineQuantizedTensor
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl

from torchao.dtypes.utils import AQTTensorImpl, is_device, Layout

aten = torch.ops.aten


@dataclass(frozen=True)
class GemlitePackedLayout(Layout):
    group_size: int = 64
    bit_width: int = 4


@register_layout(GemlitePackedLayout)
class GemliteAQTTensorImpl(TensorCoreTiledAQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else packed_weight.layout
        )
        kwargs["dtype"] = packed_weight.dtype
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout
        torch._dynamo.config.inline_inbuilt_nn_modules = False

    def __tensor_flatten__(self):
        return ["packed_weight", "scale", "zero_point"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale, zero_point = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scale"],
            tensor_data_dict["zero_point"],
        )
        (_layout,) = tensor_attributes
        return cls(packed_weight, scale, zero_point, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        from gemlite.core import DType, GemLiteLinearTriton

        group_size, bit_width = _layout.group_size, _layout.bit_width

        out_features, in_features = int_data.shape
        input_dtype, output_dtype = DType.FP16, DType.FP16
        gemlite_linear = GemLiteLinearTriton(
            bit_width,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        gemlite_linear.pack(int_data, scale, zero_point, bias=None, fma_mode=False)

        packed_weight, scale, zero_point = gemlite_linear.W_q, gemlite_linear.scales, gemlite_linear.zeros
        return cls(packed_weight, scale, zero_point, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device("cuda", device):
            raise ValueError(
                f"TensorCoreTiledAQTTensorImpl is only available for cuda device, can't convert to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale.to(device),
            self.zero_point.to(device),
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale),
            fn(self.zero_point),
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        raise NotImplementedError(
            f"GemliteAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO figure out how to do unpacking
        return self.packed_weight, self.scale, self.zero_point

    def get_layout(self) -> Layout:
        return self._layout


def _linear_fp_act_int4_weight_gemlite_check(
    input_tensor, weight_tensor, bias
):
    return (
    # input is native fp16 tensor
    not is_traceable_wrapper_subclass(input_tensor)
    and input_tensor.dtype == torch.float16
    # weight is gemlite layout
    and isinstance(weight_tensor, AffineQuantizedTensor)
    and isinstance(weight_tensor._layout, GemlitePackedLayout)
    )

def _matmul_type_fn(batch_size: int, bit_width) -> str:
    if batch_size > 64:
        return 'GEMM'
    elif batch_size > 1:
        return 'GEMM_SPLITK'
    elif bit_width<8:
        return 'GEMV_REVSPLITK'
    else:
        return 'GEMV_SPLITK'

def _linear_fp_act_int4_weight_gemlite_impl(input_tensor, weight_tensor, bias):
    from gemlite.core import GEMLITE_TRITON_MAPPING, DType
    
    batch_size = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    out_shape = input_tensor.shape[:-1] + (weight_tensor.shape[0],)
    matmul_type = _matmul_type_fn(batch_size, weight_tensor._layout.bit_width)

    out = (
        GEMLITE_TRITON_MAPPING[matmul_type]
        .forward(
            input_tensor.contiguous().view(-1, input_tensor.shape[-1]),
            weight_tensor.tensor_impl.packed_weight,
            weight_tensor.tensor_impl.scale,
            weight_tensor.tensor_impl.zero_point,
            None,
            weight_tensor._layout.bit_width,
            weight_tensor._layout.group_size,
            2**weight_tensor._layout.bit_width - 1, # 15
            32 // weight_tensor._layout.bit_width, # 8
            DType.FP16.value, # input dtype
            DType.FP16.value, # output dtype
            DType.FP16.value, # acc dtype
            DType.FP16.value, # meta_dtype
            0, # channel scale mode
            3, # group mode
            True, #data_contiguous
        )
    )
    return out.view(out_shape)
