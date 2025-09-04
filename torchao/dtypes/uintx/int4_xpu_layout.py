# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, is_device
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    fill_defaults,
)

aten = torch.ops.aten


def _aqt_is_xpu_layout_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    # TODO: use torch.uint4
    return (
        aqt.tensor_impl.dtype == torch.int32
        and aqt.quant_min == 0
        and aqt.quant_max == 15
    )


def _linear_bf16_act_uint4_weight_float_zero_check(input_tensor, weight_tensor, bias):
    return (
        # input is native bfloat16 tensor
        not is_traceable_wrapper_subclass(input_tensor)
        and input_tensor.dtype == torch.bfloat16
        and
        # weight is uint4, group quantized tensor_core_tiled tensor impl affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_xpu_layout_uint4(weight_tensor)
        and weight_tensor.dtype == torch.bfloat16
        and len(weight_tensor.shape) == 2
        and weight_tensor.zero_point_domain == ZeroPointDomain.FLOAT
        and weight_tensor.tensor_impl.scale_and_zero is not None
        and weight_tensor.tensor_impl.scale_and_zero.dtype == torch.bfloat16
        and isinstance(weight_tensor._layout, Int4XPULayout)
    )


def _linear_bf16_act_uint4_weight_float_zero_impl(input_tensor, weight_tensor, bias):
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    # TODO: check groupsize quantization
    # avoid circular dep, TODO: move this to a common util.py
    act_mat = input_tensor
    if act_mat.is_contiguous() == False:
        act_mat = act_mat.contiguous()
    # weight is packed from padded (out_features, in_features) weight tensor
    # (same dimension requirement as F.linear weight)
    packed_weight = weight_tensor.tensor_impl.packed_weight
    scales_and_zeros = weight_tensor.tensor_impl.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm(
        act_mat, packed_weight, groupsize, scales_and_zeros
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


def _linear_fp_act_uint4_weight_int8_zero_check(input_tensor, weight_tensor, bias):
    return (
        not is_traceable_wrapper_subclass(input_tensor)
        and
        # weight is uint4, group quantized tensor_core_tiled tensor impl affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_xpu_layout_uint4(weight_tensor)
        and len(weight_tensor.shape) == 2
        and weight_tensor.zero_point_domain == ZeroPointDomain.INT
        and weight_tensor.tensor_impl.scale_and_zero is None
        and weight_tensor.tensor_impl.zero.dtype == torch.int8
        and isinstance(weight_tensor._layout, Int4XPULayout)
    )


def _linear_fp_act_uint4_weight_int8_zero_impl(input_tensor, weight_tensor, bias):
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    # TODO: check groupsize quantization
    # avoid circular dep, TODO: move this to a common util.py
    act_mat = input_tensor
    # weight is packed from padded (out_features, in_features) weight tensor
    # (same dimension requirement as F.linear weight)
    packed_weight = weight_tensor.tensor_impl.packed_weight
    scale = weight_tensor.tensor_impl.scale
    zero = weight_tensor.tensor_impl.zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype
    if act_mat.numel() == 0 or packed_weight.numel() == 0:
        out_size = [s for s in orig_act_size]
        out_size[-1] = weight_tensor.size(0)
        return torch.zeros((out_size), dtype=orig_dtype, device=input_tensor.device)

    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
        act_mat, packed_weight, groupsize, scale, zero
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


@dataclass(frozen=True)
class Int4XPULayout(Layout):
    """Only for PyTorch version at least 2.7"""

    pass


@register_layout(Int4XPULayout)
class Int4XPUAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for int4 XPU layout for affine quantized tensor, this is for int4 only,
    used by tinygemm kernels `_weight_int4pack_mm_xpu` and `_weight_int4pack_mm_with_zeros_and_scales` (TBD)
    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 2-d tensor of
    dimension: [n][k / 8] (int32 dtype)
    (unpacked Tensor shape is n * k)
    Note: we also pack scale and zero point together here for tinygemm kernel
    Note: technically Int4 XPU layout should be the layout for the underlying packed weight
    (int Tensor) but since the scale and zero_point are also packed into the same tensor here which is not used
    in plain layout, we just created a layout for AQT right now, this could be improved if we split out
    int4 aqt into a separate tensor subclass
    fields:
      packed_weight (torch.Tensor): the 2-d packed tensor in a Int4 XPU layout
      [Optional] scale_and_zero (torch.Tensor): the combined scale Tensor used to map between floating point tensor to quantized tensor and zero_point Tensor
      [Optional] scale (torch.Tensor): scale tensors, should be the same dtype of packed weight
      [Optional] zeros (torch.Tensor): can be of the same dtype of packed weight or different dtype
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
        scale: torch.Tensor = None,
        zero: torch.Tensor = None,
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
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
        scale: torch.Tensor = None,
        zero: torch.Tensor = None,
    ):
        self.packed_weight = packed_weight
        self.scale_and_zero = scale_and_zero
        self.transposed = False
        self._layout = _layout
        self.scale = scale
        self.zero = zero

    def __tensor_flatten__(self):
        if self.scale_and_zero is not None:
            return ["packed_weight", "scale_and_zero"], [self.transposed, self._layout]
        else:
            return ["packed_weight", "scale", "zero"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight = tensor_data_dict["packed_weight"]
        scale_and_zero = (
            tensor_data_dict.get("scale_and_zero")
            if "scale_and_zero" in tensor_data_dict
            else None
        )
        scale = tensor_data_dict.get("scale") if "scale" in tensor_data_dict else None
        zero = tensor_data_dict.get("zero") if "zero" in tensor_data_dict else None
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(packed_weight, scale_and_zero, transposed, _layout, scale, zero)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, Int4XPULayout)
        
        def quant_2d(int_data_2d):
            if TORCH_VERSION_AT_LEAST_2_8:
                packed_weight = (int_data_2d[::, 1::2] << 4 | int_data_2d[::, ::2]).to(
                    torch.uint8
                )
                packed_weight = torch.ops.aten._convert_weight_to_int4pack(
                    packed_weight.contiguous(), 8
                )
                return packed_weight
            else:
                assert False, "INT4 not supported on XPU until 2.8"
        
        if int_data.dim() == 3:  # for moe quant
            num_experts = int_data.shape[0]
            packed_weight_list = []
            for expert in range(num_experts):
                packed_weight_list.append(quant_2d(int_data[expert]).unsqueeze(0))
            packed_weight = torch.cat(packed_weight_list, dim=0)
            scale = scale.reshape(int_data.shape[0], int_data.shape[-2], -1)
            zero_point = (
                zero_point.reshape(int_data.shape[0], int_data.shape[-2], -1)
                if zero_point is not None
                else None
            )
        else:
            assert int_data.dim() == 2
            packed_weight = quant_2d(int_data)
            scale = scale.reshape(int_data.shape[0], -1)
            zero_point = (
                zero_point.reshape(int_data.shape[0], -1)
                if zero_point is not None
                else None
            )

        if zero_point.dtype == scale.dtype:
            from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

            scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point)
            return cls(packed_weight, scale_and_zero, False, _layout, None, None)
        else:
            return cls(
                packed_weight,
                None,
                False,
                _layout,
                scale.transpose(-2, -1).contiguous(),
                zero_point.transpose(-2, -1).contiguous().to(torch.int8),
            )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"Int4XPUAQTTensorImpl does not support conversion from {self.device} to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device) if self.scale_and_zero is not None else None,
            self.transposed,
            self._layout,
            self.scale.to(device) if self.scale is not None else None,
            self.zero.to(device) if self.zero is not None else None,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale_and_zero) if self.scale_and_zero is not None else None,
            self.transposed,
            self._layout,
            fn(self.scale) if self.scale is not None else None,
            fn(self.zero) if self.zero is not None else None,
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
         
        if func in [aten.select.int, aten.index.Tensor]:
            assert not (func is aten.select.int and args[1] != 0), (
                "aten.select.int currently only has support for dim=0"
            )
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
            )

        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = Int4XPUAQTTensorImpl(
                args[0].packed_weight,
                args[0].scale_and_zero,
                not args[0].transposed,
                args[0]._layout,
                args[0].scale,
                args[0].zero,
            )
            return return_and_correct_aliasing(func, args, kwargs, transposed)

        if func is torch.ops.aten.copy_.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                int_data, scale, zero_point = self.get_plain()
                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                int_data = self._layout.post_process(int_data)
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            elif dim == 1:
                int_data, scale, zero_point = self.get_plain()
                assert step == 1, "Only step == 1 is supported in slicing right now"
                data_len = int_data.shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                # this is to handle padding
                int_data = self._layout.post_process(int_data)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                zero_point = aten.slice.Tensor(
                    zero_point, dim, start_scale, end_scale, step
                )
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return sliced
            else:
                raise NotImplementedError(
                    f"Int4XPUAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"Int4XPUAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchao.quantization.quant_primitives import (
            _quantize_affine_tinygemm,
            quantize_affine,
        )
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

        if self.scale_and_zero is not None:
            scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)
        else:
            scale = self.scale.transpose(0, 1).contiguous()
            zero = self.zero.transpose(0, 1).contiguous()

        cur_shape = self.shape
        assert len(cur_shape) == 2
        original_shape = (cur_shape[0], cur_shape[1] * 8)
        eye_shape = original_shape[1]
        groupsize = int(original_shape[1] / scale.shape[1])
        block_size = (1, groupsize)
        device = self.device
        original_dtype = torch.bfloat16
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        assert len(block_size) == 2 and block_size[0] == 1
        if self.scale_and_zero is None:
            dequantized = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                torch.eye(eye_shape, device=device, dtype=original_dtype),
                self.packed_weight,
                groupsize,
                self.scale,
                self.zero,
            )
            dequantized = dequantized.t().contiguous()
            int_data = quantize_affine(
                dequantized,
                block_size,
                scale,
                zero,
                target_dtype,
                quant_min,
                quant_max,
            )
        else:
            dequantized = torch.ops.aten._weight_int4pack_mm(
                torch.eye(eye_shape, device=device, dtype=original_dtype),
                self.packed_weight,
                groupsize,
                self.scale_and_zero,
            )
            dequantized = dequantized.t().contiguous()
            # TODO: move this to `unpack_tinygemm_scales_and_zeros`?
            scale = scale.reshape(scale.shape[:-1]).contiguous()
            zero = zero.reshape(zero.shape[:-1]).contiguous()
            int_data = _quantize_affine_tinygemm(
                dequantized,
                block_size,
                scale,
                zero,
                target_dtype,
                quant_min,
                quant_max,
            )
        return int_data, scale, zero

    def get_layout(self) -> Layout:
        return self._layout
