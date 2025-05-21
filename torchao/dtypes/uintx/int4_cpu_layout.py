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
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    quantize_affine_float_zero_point,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    fill_defaults,
)

aten = torch.ops.aten


@dataclass(frozen=True)
class Int4CPULayout(Layout):
    """Layout class for int4 CPU layout for affine quantized tensor, used by tinygemm kernels `_weight_int4pack_mm_for_cpu`.
    Only for PyTorch version at least 2.6
    """

    pass


@register_layout(Int4CPULayout)
class Int4CPUAQTTensorImpl(AQTTensorImpl):
    """TensorImpl for int4 CPU layout for affine quantized tensor, this is for int4 only,
    used by tinygemm kernels `_weight_int4pack_mm_for_cpu`
    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 2-d tensor of
    dimension: [n][k / 2] (uint8 dtype)
    (unpacked Tensor shape is n * k)
    Note: we also pack scale and zero point together here for tinygemm kernel
    Note: technically Int4 CPU layout should be the layout for the underlying packed weight
    (int Tensor) but since the scale and zero_point are also packed into the same tensor here which is not used
    in plain layout, we just created a layout for AQT right now, this could be improved if we split out
    int4 aqt into a separate tensor subclass
    fields:
      packed_weight (torch.Tensor): the 2-d packed tensor in a Int4 CPU layout
      scale_and_zero (torch.Tensor): the combined scale Tensor used to map between floating point tensor to quantized tensor and zero_point Tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale_and_zero: torch.Tensor,
        transposed: bool,
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
        scale_and_zero: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale_and_zero = scale_and_zero
        self.transposed = False
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_weight", "scale_and_zero"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale_and_zero = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scale_and_zero"],
        )
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(packed_weight, scale_and_zero, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, Int4CPULayout)

        if TORCH_VERSION_AT_LEAST_2_6:
            assert int_data.dtype == torch.int32, (
                "torch.ops.aten._convert_weight_to_int4pack_for_cpu expects `int32` dtype"
            )
            packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                int_data,
                1,  # TODO:remove
            )
        elif TORCH_VERSION_AT_LEAST_2_5:
            int_data = (int_data[::, ::2] << 4 | int_data[::, 1::2]).to(torch.uint8)
            assert int_data.dtype == torch.uint8, (
                "torch.ops.aten._convert_weight_to_int4pack in torch 2.5 expects `uint8` dtype"
            )
            packed_weight = torch.ops.aten._convert_weight_to_int4pack(
                int_data, _layout.inner_k_tiles
            )
        else:
            assert int_data.dtype == torch.int32, (
                "torch.ops.aten._convert_weight_to_int4pack in torch 2.4 expects `int32` dtype"
            )
            packed_weight = torch.ops.aten._convert_weight_to_int4pack(
                int_data, _layout.inner_k_tiles
            )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)
        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point, scale.dtype)
        return cls(packed_weight, scale_and_zero, False, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"Int4CPUAQTTensorImpl does not support conversion from {self.device} to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device),
            self.transposed,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale_and_zero),
            self.transposed,
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

        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = Int4CPUAQTTensorImpl(
                args[0].packed_weight,
                args[0].scale_and_zero,
                not args[0].transposed,
                args[0]._layout,
            )
            return return_and_correct_aliasing(func, args, kwargs, transposed)

        if func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim in [0, 1]:
                assert step == 1, "Only step == 1 is supported in slicing right now"
                int_data, scale, zero_point = self.get_plain()
                data_len = int_data.shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                zero_point = aten.slice.Tensor(
                    zero_point, dim, start_scale, end_scale, step
                )
                # this is to handle padding
                int_data, scale, zero_point = self._layout.post_process(
                    int_data, scale, zero_point, self.block_size
                )
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            else:
                raise NotImplementedError(
                    f"Int4CPUAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"Int4CPUAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def block_size(self):
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

        scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)
        cur_shape = self.shape
        assert len(cur_shape) == 4
        inner_k_tiles = cur_shape[-1] * 2
        original_shape = (cur_shape[0] * 8, cur_shape[1] * (inner_k_tiles * 16))
        groupsize = int(original_shape[1] / scale.shape[-2])
        return (1, groupsize)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchao.quantization.utils import unpack_tinygemm_scales_and_zeros

        scale, zero = unpack_tinygemm_scales_and_zeros(self.scale_and_zero)

        cur_shape = self.shape
        assert len(cur_shape) == 2
        original_shape = (cur_shape[0], cur_shape[1] * 2)
        eye_shape = original_shape[1]
        groupsize = int(original_shape[1] / scale.shape[-2])
        block_size = (1, groupsize)
        device = self.device
        original_dtype = self.scale_and_zero.dtype
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        # zero_point_domain is ZeroPointDomain.FLOAT  # TODO: clean up later
        assert len(block_size) == 2 and block_size[0] == 1
        dequantized = torch.ops.aten._weight_int4pack_mm_for_cpu(
            torch.eye(eye_shape, device=device, dtype=original_dtype),
            self.packed_weight,
            groupsize,
            self.scale_and_zero,
        )
        dequantized = dequantized.t().contiguous()
        # TODO: move this to `unpack_tinygemm_scales_and_zeros`?
        scale = scale.reshape(scale.shape[:-1]).contiguous()
        zero = zero.reshape(zero.shape[:-1]).contiguous()
        int_data = quantize_affine_float_zero_point(
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


def _aqt_is_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.uint8
        and aqt.quant_min == 0
        and aqt.quant_max == 15
    )


def _is_float(dtype):
    return dtype in (torch.float, torch.half, torch.bfloat16)


def _linear_fp_act_uint4_weight_cpu_check(input_tensor, weight_tensor, bias):
    return (
        TORCH_VERSION_AT_LEAST_2_6
        and is_device(input_tensor.device.type, "cpu")
        and is_device(weight_tensor.device.type, "cpu")
        and (bias is None or is_device(bias.device.type, "cpu"))
        and not is_traceable_wrapper_subclass(input_tensor)
        and _is_float(input_tensor.dtype)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_uint4(weight_tensor)
        and _is_float(weight_tensor.dtype)
        and len(weight_tensor.shape) == 2
        and weight_tensor.zero_point_domain == ZeroPointDomain.FLOAT
        and isinstance(weight_tensor._layout, Int4CPULayout)
    )


def _linear_fp_act_uint4_weight_cpu_impl(input_tensor, weight_tensor, bias):
    assert TORCH_VERSION_AT_LEAST_2_6, (
        f"Requires PyTorch version at least 2.6, but got: {torch.__version__}"
    )
    assert is_device(input_tensor.device.type, "cpu"), (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    act_mat = input_tensor
    packed_weight = weight_tensor.tensor_impl.packed_weight
    scale_and_zero = weight_tensor.tensor_impl.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm_for_cpu(
        act_mat.contiguous(), packed_weight, groupsize, scale_and_zero
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)
