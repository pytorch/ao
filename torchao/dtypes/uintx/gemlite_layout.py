# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
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
from torchao.dtypes.utils import Layout
from torchao.utils import fill_defaults

try:
    import gemlite
except:
    gemlite = None

aten = torch.ops.aten


def _same_metadata(
    self: "GemliteAQTTensorImpl",
    src: "GemliteAQTTensorImpl",
) -> bool:
    kwargs_match = len(self.gemlite_kwargs) == len(src.gemlite_kwargs)
    for k, v in self.gemlite_kwargs.items():
        kwargs_match = kwargs_match and (v == src.gemlite_kwargs[k])

    return (
        isinstance(self, GemliteAQTTensorImpl)
        and isinstance(src, GemliteAQTTensorImpl)
        and self.shape == src.shape
        and self.packed_weight.shape == src.packed_weight.shape
        and self.scale.shape == src.scale.shape
        and self.zero_point.shape == src.zero_point.shape
        and kwargs_match
        and type(self._layout) == type(src._layout)
    )


def get_gemlite_quant_kwargs(bit_width, group_size, dtype):
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
        kwargs["zero_point_dtype"] = dtype
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
    packing_bitwidth=None,
    use_hqq=True,
):
    if gemlite is None:
        raise ImportError(
            "Unable to import 'gemlite'. Please ensure it is installed correctly. You can install it with: pip install gemlite"
        )

    assert bit_width in [
        4,
        8,
    ], f"gemlite only works with bit_width 4,8 but got {bit_width}"

    assert weight.dtype in [torch.float16, torch.bfloat16], (
        f"gemlite only works with dtype torch.float16 or torch.bfloat16 but got {weight.dtype}"
    )
    assert group_size in [32, 64, 128, 256, 512, 1024, None]
    assert group_size is None or bit_width != 8, (
        "gemlite only works with group_size=None for bit_width=8"
    )
    assert packing_bitwidth in [8, 16, 32, None], (
        f"Invalid packing bitwidth, got {packing_bitwidth}"
    )

    out_features, in_features = weight.shape
    group_size = in_features if group_size is None else group_size

    aqt_kwargs = get_gemlite_quant_kwargs(bit_width, group_size, weight.dtype)
    aqt_kwargs["_layout"] = GemlitePackedLayout(
        group_size=group_size,
        bit_width=bit_width,
        packing_bitwidth=packing_bitwidth,
    )
    aqt_kwargs["use_hqq"] = use_hqq
    return aqt_kwargs


@dataclass(frozen=True)
class GemlitePackedLayout(Layout):
    group_size: Optional[int] = 128
    bit_width: int = 4
    packing_bitwidth: Optional[int] = None


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
        assert isinstance(_layout, GemlitePackedLayout), (
            f"GemliteAQTTensorImpl only works with GemliteLinearTriton but got {_layout}"
        )
        device = int_data.device
        if device.type != "cuda":
            int_data = (
                int_data.cuda()
            )  # We need int_data on cuda device because of Triton packing

        group_size, bit_width = _layout.group_size, _layout.bit_width
        out_features, in_features = int_data.shape
        packing_bitwidth = _layout.packing_bitwidth

        if bit_width == 8 and group_size == in_features:
            gemlite_linear = gemlite.helper.A16W8(device=int_data.device).from_weights(
                int_data, scales=scale, bias=None
            )
        else:
            gemlite_linear = gemlite.helper.A16Wn(
                device=int_data.device, packing_bitwidth=packing_bitwidth
            ).from_weights(
                int_data, scale, zero_point, bit_width, group_size, bias=None
            )

        meta_args = gemlite_linear.get_meta_args()
        gemlite_kwargs = {
            "in_features": in_features,
            "out_features": out_features,
            "packing_bitwidth": packing_bitwidth,
            "data_contiguous": gemlite_linear.data_contiguous,
            "elements_per_sample": gemlite_linear.elements_per_sample,
            "W_group_mode": gemlite_linear.W_group_mode,
            "meta_args": meta_args,
        }

        packed_weight, scale, zero_point = gemlite_linear.get_tensor_args()
        packed_weight = packed_weight.to(device)

        return cls(packed_weight, scale, zero_point, gemlite_kwargs, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
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
        device = self.packed_weight.device
        int_data = (
            (
                gemlite.bitpack.unpack_over_rows(
                    self.packed_weight.cuda(),
                    W_nbits=self._layout.bit_width,
                    num_output_rows=self.gemlite_kwargs["in_features"],
                    dtype=torch.uint8,
                )
            )
            .to(device)
            .t()
        )

        # Preserve col-row major layout
        if self.gemlite_kwargs["data_contiguous"]:
            int_data = int_data.contiguous()

        # Handle FMA mode: W_q * s + z  -> (W_q - z) * s
        if self.gemlite_kwargs["W_group_mode"] == 4:
            scale_min_val = 1e-8
            scale = self.scale.clone().float()
            scale[torch.logical_and(scale >= 0, scale.abs() <= scale_min_val)] = (
                scale_min_val
            )
            scale[
                torch.logical_and(scale < 0, scale.abs() <= scale_min_val)
            ] = -scale_min_val
            zero_point = (-self.zero_point.float() / scale).clamp_(-100, 100)
            zero_point = zero_point.to(self.scale.dtype)
        else:
            zero_point = self.zero_point

        scale = self.scale.t().contiguous()
        zero_point = zero_point.t().contiguous()

        return int_data, scale, zero_point

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
            assert step == 1, "Only step == 1 is supported in slicing right now"

            if dim in [0, 1]:
                # data in self is transposed, meaning forward() performs x @ W_deq not x @ W_deq.T
                dim = 1 - dim
                packed_weight = self.packed_weight
                scale = self.scale
                zero_point = self.zero_point

                gemlite_kwargs = self.gemlite_kwargs.copy()
                orig_shape = [
                    gemlite_kwargs["in_features"],
                    gemlite_kwargs["out_features"],
                ]
                elements_per_sample = gemlite_kwargs["elements_per_sample"]
                data_len = orig_shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                # For packing only the K dimension. This should be flipped for N-dim packing.
                div = elements_per_sample if dim == 0 else 1
                packed_weight = aten.slice.Tensor(
                    packed_weight, dim, start // div, end // div, step
                )

                # Update in_features/out_features
                gemlite_kwargs["in_features"] = (
                    packed_weight.shape[0] * elements_per_sample
                )
                gemlite_kwargs["out_features"] = packed_weight.shape[1]

                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                if zero_point is not None and zero_point.numel() > 0:
                    zero_point = aten.slice.Tensor(
                        zero_point, dim, start_scale, end_scale, step
                    )
                else:
                    zero_point = None

                sliced = GemliteAQTTensorImpl(
                    packed_weight, scale, zero_point, gemlite_kwargs, self._layout
                )
                return return_and_correct_aliasing(func, args, kwargs, sliced)

            else:
                raise NotImplementedError(
                    f"GemliteAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        elif func is aten.copy_.default:
            self = args[0]
            src = args[1]
            if _same_metadata(self, src):
                self_tensors = self.__tensor_flatten__()[0]
                for tensor_name in self_tensors:
                    getattr(self, tensor_name).copy_(getattr(src, tensor_name))
                return
            raise ValueError(
                f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
            )

        raise NotImplementedError(
            f"GemliteAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_layout(self) -> Layout:
        return self._layout

    @property
    def block_size(self):
        return (1, self._layout.group_size)


def _linear_fp_act_int4_weight_gemlite_impl(input_tensor, weight_tensor, bias=None):
    if hasattr(weight_tensor, "tensor_impl"):
        weight_impl = weight_tensor.tensor_impl
    else:
        weight_impl = weight_tensor

    return gemlite.core.forward_functional(
        x=input_tensor,
        bias=bias,
        tensor_args=(
            weight_impl.packed_weight,
            weight_impl.scale,
            weight_impl.zero_point,
        ),
        meta_args=weight_impl.gemlite_kwargs["meta_args"],
    )


def _linear_fp_act_int4_weight_gemlite_check(input_tensor, weight_tensor, bias):
    return (
        # input is native fp16 tensor
        not is_traceable_wrapper_subclass(input_tensor)
        # and input_tensor.dtype in [torch.float16, torch.bfloat16]
        # weight is gemlite layout
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and isinstance(weight_tensor._layout, GemlitePackedLayout)
    )
