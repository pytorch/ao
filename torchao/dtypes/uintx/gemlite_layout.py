# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl
from torchao.dtypes.utils import Layout, is_device
from torchao.quantization.quant_primitives import quantize_affine
from torchao.utils import fill_defaults

aten = torch.ops.aten


def get_gemlite_quant_kwargs(bit_width, group_size):
    from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain

    kwargs = {}
    if bit_width != 8:
        kwargs["mapping_type"] = MappingType.ASYMMETRIC
        kwargs["block_size"] = (1, group_size)
        kwargs["target_dtype"] = torch.uint8
        kwargs["eps"] = 1e-6
        kwargs["quant_min"] = 0
        kwargs["quant_max"] = (2**bit_width) - 1
        kwargs["eps"] = 1e-6
        kwargs["zero_point_dtype"] = torch.float16
        kwargs["zero_point_domain"] = ZeroPointDomain.FLOAT
    elif bit_width == 8:
        kwargs["mapping_type"] = MappingType.SYMMETRIC
        kwargs["block_size"] = (1, group_size)
        kwargs["target_dtype"] = torch.int8
        kwargs["quant_min"] = -128
        kwargs["quant_max"] = 127
        kwargs["eps"] = 1e-5
        kwargs["zero_point_dtype"] = None
        kwargs["zero_point_domain"] = ZeroPointDomain.NONE
    return kwargs


def get_gemlite_aqt_kwargs(
    weight,
    group_size=64,
    bit_width=4,
    packing_bitwidth=32,
    contiguous=None,
    use_hqq=True,
):
    from torchao.dtypes.uintx.gemlite_layout import GemlitePackedLayout

    assert bit_width in [
        4,
        8,
    ], f"gemlite only works with bit_width 4,8 but got {bit_width}"
    assert packing_bitwidth in [
        8,
        16,
        32,
    ], f"gemlite needs packing_bitwidth in [8, 16, 32] but got {packing_bitwidth}"
    assert (
        weight.dtype == torch.float16
    ), f"gemlite only works with dtype torch.float16 but got {weight.dtype}"
    assert group_size in [32, 64, 128, 256, 512, 1024, None]
    assert (
        group_size is None or bit_width != 8
    ), "gemlite only works with group_size=None for bit_width=8"

    out_features, in_features = weight.shape
    group_size = in_features if group_size is None else group_size

    if in_features % 128 != 0 and out_features % 128 != 0:
        warnings.simplefilter("once", UserWarning)
        warnings.warn(
            "Gemlite only works for layers with in_features or out_features divisible by 128, "
            + "some layers have been skipped",
            UserWarning,
        )
        return weight

    aqt_kwargs = get_gemlite_quant_kwargs(bit_width, group_size)
    aqt_kwargs["_layout"] = GemlitePackedLayout(
        group_size=group_size,
        bit_width=bit_width,
        packing_bitwidth=packing_bitwidth,
        contiguous=contiguous,
    )
    aqt_kwargs["use_hqq"] = use_hqq
    return aqt_kwargs


@dataclass(frozen=True)
class GemlitePackedLayout(Layout):
    group_size: Optional[int] = 64
    bit_width: int = 4
    packing_bitwidth: int = 8
    contiguous: bool = None


@register_layout(GemlitePackedLayout)
class GemliteAQTTensorImpl(TensorCoreTiledAQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        gemlite_kwargs: Dict,
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
        gemlite_kwargs: Dict,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self.gemlite_kwargs = gemlite_kwargs
        self._layout = _layout
        torch._dynamo.config.inline_inbuilt_nn_modules = False

    def __tensor_flatten__(self):
        return ["packed_weight", "scale", "zero_point"], [
            self._layout,
            self.gemlite_kwargs,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale, zero_point = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scale"],
            tensor_data_dict["zero_point"],
        )
        _layout, gemlite_kwargs = tensor_attributes
        return cls(packed_weight, scale, zero_point, gemlite_kwargs, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        from gemlite.core import DType, GemLiteLinearTriton, set_autotune

        assert isinstance(
            _layout, GemlitePackedLayout
        ), f"GemliteAQTTensorImpl only works with GemliteLinearTriton but got {_layout}"
        group_size, bit_width = _layout.group_size, _layout.bit_width

        torch._dynamo.config.inline_inbuilt_nn_modules = False
        set_autotune(
            {"GEMV_REVSPLITK": True, "GEMV": True, "GEMM_SPLITK": True, "GEMM": True},
            exhaustive=False,
            use_cuda_graph=False,
        )
        if _layout.group_size is None and _layout.bit_width == 4:
            from gemlite.core import GEMLITE_ACC_DTYPE
            from gemlite.dtypes import DType

            GEMLITE_ACC_DTYPE[DType.FP16] = DType.FP32

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
        gemlite_linear.pack(
            int_data,
            scale,
            zero_point,
            bias=None,
            fma_mode=False,
            packing_bitwidth=_layout.packing_bitwidth,
            contiguous=_layout.contiguous,
        )

        gemlite_kwargs = {
            "out_features": out_features,
            "scale_activations": gemlite_linear.scale_activations,
            "meta_args": gemlite_linear.get_meta_args(),
        }

        packed_weight, scale, zero_point = gemlite_linear.get_tensor_args()

        return cls(packed_weight, scale, zero_point, gemlite_kwargs, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device("cuda", device):
            raise ValueError(
                f"GemliteAQTTensorImpl is only available for cuda device, can't convert to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale.to(device),
            self.zero_point.to(device),
            self.gemlite_kwargs,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale),
            fn(self.zero_point),
            self.gemlite_kwargs,
            self._layout,
        )

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq = (
            _linear_fp_act_int4_weight_gemlite_impl(
                torch.eye(
                    self.scale.shape[0] * self._layout.group_size,
                    device=self.device,
                    dtype=self.scale.dtype,
                ),
                self,
            )
            .t()
            .contiguous()
        )

        quant_kwargs = get_gemlite_quant_kwargs(
            self._layout.bit_width, self._layout.group_size
        )
        quant_kwargs["output_dtype"] = quant_kwargs.pop("target_dtype")
        for key in ["mapping_type", "eps", "zero_point_dtype"]:
            del quant_kwargs[key]

        int_data = quantize_affine(
            dq,
            scale=self.scale,
            zero_point=self.zero_point,
            **quant_kwargs,
        )

        return int_data, self.scale, self.zero_point

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        # we don't handle transpose operations and just ignore them. In practice the only
        # reason a transpsoe should occur is because the functional linear
        # op can decompose into e.g. transpose + addmm so since we want
        # to use the gemlite matmul kernel, which expects teh weight to be passed in as is,
        # we ignore the transpose
        if func is aten.detach.default or func is aten.t.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                int_data, scale, zero_point = self.get_plain()
                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                int_data = self._layout.post_process(int_data)
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            elif dim == 1:
                int_data, scale, zero_point = self.get_plain()
                assert step == 1, "Only step == 1 is supported in slicing right now"
                data_len = int_data.shape[dim]
                # scale and zero_point are transposed compared to int_data
                param_dim = 1 - dim
                scale_len = scale.shape[param_dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                scale = aten.slice.Tensor(
                    scale, param_dim, start_scale, end_scale, step
                )
                if zero_point is not None and zero_point.numel() > 0:
                    zero_point = aten.slice.Tensor(
                        zero_point, param_dim, start_scale, end_scale, step
                    )
                else:
                    zero_point = None
                # import fbvscode; fbvscode.set_trace()
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return sliced
            else:
                raise NotImplementedError(
                    f"GemliteAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"GemliteAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_layout(self) -> Layout:
        return self._layout


# logic taken from gemlite's core.py
def _matmul_type_fn(batch_size: int, bit_width: int) -> str:
    if batch_size > 64:
        return "GEMM"
    elif batch_size > 1:
        return "GEMM_SPLITK"
    elif bit_width < 8:
        return "GEMV_REVSPLITK"
    else:
        return "GEMV_SPLITK"


def _linear_fp_act_int4_weight_gemlite_impl(input_tensor, weight_tensor, bias=None):
    if hasattr(weight_tensor, "tensor_impl"):
        weight_impl = weight_tensor.tensor_impl
    else:
        weight_impl = weight_tensor

    from gemlite.core import GemLiteLinearTriton

    batch_size = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    matmul_type = _matmul_type_fn(batch_size, weight_impl._layout.bit_width)

    return GemLiteLinearTriton.forward_functional(
        x=input_tensor,
        bias=bias,
        matmul_type=matmul_type,
        **weight_impl.gemlite_kwargs,
        tensor_args=(
            weight_impl.packed_weight,
            weight_impl.scale,
            weight_impl.zero_point,
        ),
    )


def _linear_fp_act_int4_weight_gemlite_check(input_tensor, weight_tensor, bias):
    return (
        # input is native fp16 tensor
        not is_traceable_wrapper_subclass(input_tensor)
        # and input_tensor.dtype == torch.float16
        # weight is gemlite layout
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor._layout, GemlitePackedLayout)
    )
