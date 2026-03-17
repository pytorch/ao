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
from torchao.utils import fill_defaults

aten = torch.ops.aten


def _aqt_is_npu_layout_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized with NPU layout"""
    return (
        aqt.tensor_impl.dtype == torch.int32
        and aqt.quant_min == 0
        and aqt.quant_max == 15
    )


def _linear_bf16_act_uint4_weight_npu_check(input_tensor, weight_tensor, bias):
    from torchao.quantization.quant_primitives import ZeroPointDomain

    return (
        # input is native bfloat16 tensor
        not is_traceable_wrapper_subclass(input_tensor)
        and input_tensor.dtype == torch.bfloat16
        and is_device(input_tensor.device.type, "npu")
        and
        # weight is uint4 affine quantized tensor with NPU layout
        isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_npu_layout_uint4(weight_tensor)
        and weight_tensor.dtype == torch.bfloat16
        and len(weight_tensor.shape) == 2
        and weight_tensor.zero_point_domain == ZeroPointDomain.FLOAT
        and weight_tensor.tensor_impl.scale_and_zero is not None
        and weight_tensor.tensor_impl.scale_and_zero.dtype == torch.bfloat16
        and isinstance(weight_tensor._layout, Int4NPULayout)
    )


def _linear_bf16_act_uint4_weight_npu_impl(input_tensor, weight_tensor, bias):
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    act_mat = input_tensor
    if not act_mat.is_contiguous():
        act_mat = act_mat.contiguous()

    packed_weight = weight_tensor.tensor_impl.packed_weight
    scales_and_zeros = weight_tensor.tensor_impl.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    act_mat = act_mat.reshape(-1, act_mat.shape[-1]).to(torch.bfloat16)

    groupsize = weight_tensor.block_size[1]

    try:
        import torch_npu

        # NOTE: torch_npu op name and signature need to be verified against
        # the actual torch_npu documentation / release version being used.
        y = torch_npu.npu_weight_quant_batchmatmul(
            act_mat, packed_weight, groupsize, scales_and_zeros
        )
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "torch_npu is required for NPU Int4 quantized linear. "
            f"Original error: {e}"
        )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


@dataclass(frozen=True)
class Int4NPULayout(Layout):
    """Int4 layout for Huawei Ascend NPU via torch_npu."""

    pass


@register_layout(Int4NPULayout)
class Int4NPUAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for int4 NPU layout for affine quantized tensor, targeting
    Huawei Ascend NPU via torch_npu.

    It stores the original tensor of dimension [n][k] (int32 dtype) as a
    packed weight tensor suitable for NPU int4 matmul kernels.

    fields:
      packed_weight (torch.Tensor): the packed tensor in Int4 NPU layout
      scale_and_zero (torch.Tensor): the combined scale/zero_point tensor (bf16)
      transposed (bool): whether the weight is transposed
      _layout (Layout): the Int4NPULayout instance
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
        if self.scale_and_zero is not None:
            return ["packed_weight", "scale_and_zero"], [self.transposed, self._layout]
        else:
            return ["packed_weight"], [self.transposed, self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight = tensor_data_dict["packed_weight"]
        scale_and_zero = tensor_data_dict.get("scale_and_zero", None)
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
        assert isinstance(_layout, Int4NPULayout)
        assert int_data.dtype == torch.int32, (
            "Int4NPUAQTTensorImpl expects int32 dtype for int_data"
        )

        # Try NPU-specific weight packing op, fall back to manual bit packing
        try:
            if hasattr(torch.ops, "npu") and hasattr(
                torch.ops.npu, "npu_convert_weight_to_int4pack"
            ):
                packed_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                    int_data.contiguous()
                )
            else:
                # Manual bit packing: pack two int4 values into one uint8
                packed_weight = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(
                    torch.uint8
                )
        except Exception:
            packed_weight = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(
                torch.uint8
            )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)

        from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point)
        return cls(packed_weight, scale_and_zero, False, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs["device"]
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"Int4NPUAQTTensorImpl does not support conversion from {self.device} to {device}"
            )
        return self.__class__(
            self.packed_weight.to(device),
            self.scale_and_zero.to(device) if self.scale_and_zero is not None else None,
            self.transposed,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale_and_zero) if self.scale_and_zero is not None else None,
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
            transposed = Int4NPUAQTTensorImpl(
                args[0].packed_weight,
                args[0].scale_and_zero,
                not args[0].transposed,
                args[0]._layout,
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
                int_data = self._layout.post_process(int_data)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                zero_point = aten.slice.Tensor(
                    zero_point, dim, start_scale, end_scale, step
                )
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return sliced
            else:
                raise NotImplementedError(
                    f"Int4NPUAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"Int4NPUAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Int4NPUAQTTensorImpl.get_plain() is not implemented. "
            "NPU-specific dequantization requires torch_npu ops. "
            "This may be needed for operations like tensor slicing."
        )

    def get_layout(self) -> Layout:
        return self._layout
