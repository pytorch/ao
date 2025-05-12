# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    get_tensor_impl_constructor,
    register_layout,
)
from torchao.dtypes.uintx.plain_layout import (
    _aqt_is_int8_reduced_range,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    choose_qparams_and_quantize_affine_qqq,
    dequantize_affine_qqq,
)

logger = logging.getLogger(__name__)

aten = torch.ops.aten


class MarlinQQQTensor(AffineQuantizedTensor):
    """MarlinQQQ quantized tensor subclass which inherits AffineQuantizedTensor class.

    To see what happens during choose_qparams_and_quantize_affine_qqq, quantization and dequantization for marlin qqq quantization,
    please checkout https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py
    and check the two quant primitive ops: choose_qparams_and_quantize_affine_qqq and dequantize_affine_qqq
    """

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        int_data, s_group, s_channel = self.tensor_impl.get_plain()
        nbits = int(math.log2(self.quant_max - self.quant_min + 1))
        group_size = max(self.block_size)
        return dequantize_affine_qqq(
            int_data, s_group, s_channel, nbits, group_size, output_dtype
        )

    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        block_size: Tuple[int, ...],
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        _layout: Optional[Layout] = None,
    ):
        """Converts a floating point tensor to a Marlin QQQ quantized tensor."""
        if zero_point_domain is None:
            raise ValueError("Please use ZeroPointDomain.NONE instead of None")
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)
        nbits = int(math.log2(quant_max - quant_min + 1))
        group_size = max(block_size)
        data, s_group, s_channel, _ = choose_qparams_and_quantize_affine_qqq(
            input_float, nbits, group_size
        )
        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(data, s_group, s_channel, _layout)
        return cls(
            tensor_impl,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype,
        )


@dataclass(frozen=True)
class MarlinQQQLayout(Layout):
    """MarlinQQQLayout is a layout class for Marlin QQQ quantization."""

    pass


@register_layout(MarlinQQQLayout)
class MarlinQQQAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl storage class for sparse_qqq layout for affine quantized tensor.

    Can only be used with 4 bits quantization for now.

    Original marlin documentation and information:
    https://github.com/IST-DASLab/marlin/tree/master

    Marlin qqq information:
    https://github.com/HandH1998/QQQ/tree/main
    https://arxiv.org/pdf/2406.09904

    fields:
        original_shape (torch.Size): the original shape of the tensor. used to unpack the tensor to the original shape
        group_size (int): the group size used to pack the tensor
        num_bits (int): the number of bits used to quantize the tensor
    """

    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        _layout: Layout,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        self.int_data = int_data
        self.s_group = s_group
        self.s_channel = s_channel
        self._layout = _layout
        self.original_shape = original_shape
        self.group_size = group_size
        self.num_bits = num_bits

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"MarlinQQQAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["int_data", "s_group", "s_channel"], [
            self._layout,
            self.original_shape,
            self.group_size,
            self.num_bits,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data = tensor_data_dict["int_data"]
        s_group = tensor_data_dict["s_group"]
        s_channel = tensor_data_dict["s_channel"]
        _layout, original_shape, group_size, num_bits = tensor_attributes
        return cls(
            int_data, s_group, s_channel, _layout, original_shape, group_size, num_bits
        )

    def get_plain(self):
        from torchao.quantization.marlin_qqq import (
            unpack_from_marlin_qqq,
        )

        int_data_expanded, s_group_expanded, s_channel_expanded = (
            unpack_from_marlin_qqq(
                self.int_data,
                self.s_group,
                self.s_channel,
                self.original_shape,
                self.num_bits,
                self.group_size,
            )
        )
        int_data_expanded_t = int_data_expanded.t()
        s_group_expanded_t = s_group_expanded.t()
        s_channel_expanded_t = s_channel_expanded.t()
        return int_data_expanded_t, s_group_expanded_t, s_channel_expanded_t

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        _layout: Layout,
    ):
        from torchao.quantization.marlin_qqq import (
            const,
            pack_to_marlin_qqq,
        )

        assert isinstance(_layout, MarlinQQQLayout)

        # Linear layers are (in_features, out_features) but the int_data that is reaching this point
        # is (out_features, in_features). We need to transpose it to match the expected shape in the marlin code.
        q_w = int_data.t()
        s_group_t = s_group.t()
        s_channel_t = s_channel.t()

        if not torch.cuda.get_device_capability()[0] >= 8:
            raise ValueError(
                f"Can not use Marlin QQQ int4*int8 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel."
            )

        if q_w.dtype != torch.int32:
            raise ValueError("Only `torch.int32` weights are supported.")

        in_features, out_features = q_w.shape
        # (thread_k, thread_n)
        thread_config = [(64, 256), (128, 128), (128, 64), (64, 128)]
        if not any(
            [
                in_features % thread_k == 0 and out_features % thread_n == 0
                for thread_k, thread_n in thread_config
            ]
        ):
            raise ValueError(
                "Not supported `in_features`: {} and `out_features`: {}.".format(
                    in_features, out_features
                )
            )

        num_bits = 4 if torch.max(q_w) - torch.min(q_w) < 16 else -1
        if num_bits not in [4]:
            raise ValueError(f"Only {[4]} bits are supported, got {num_bits}.")

        if s_group.numel() == 0:
            group_size = -1
        else:
            group_size = in_features // s_group_t.shape[0]
        assert group_size <= in_features, (
            "Group size must be less than or equal to in_features."
        )

        if group_size not in const.SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Only {const.SUPPORTED_GROUP_SIZES} group sizes are supported, got {group_size}."
            )

        # Compress quantized weight to marlin format
        marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel = pack_to_marlin_qqq(
            q_w, s_group_t, s_channel_t, num_bits, group_size
        )

        return cls(
            marlin_qqq_q_w,
            marlin_qqq_s_group,
            marlin_qqq_s_channel,
            _layout,
            q_w.shape,
            group_size,
            num_bits,
        )

    def get_layout(self) -> Layout:
        return self._layout

    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.s_group = fn(self.s_group)
        self.s_channel = fn(self.s_channel)
        return self


def _linear_int8_act_int4_weight_marlin_qqq_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and input_tensor.dtype == torch.float16
        and input_tensor.tensor_impl.scale.dtype == torch.float32
        and len(input_tensor.tensor_impl.scale.shape) == len(input_tensor.shape) - 1
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and weight_tensor.tensor_impl.dtype == torch.int32
        and len(weight_tensor.shape) == 2
        and isinstance(weight_tensor._layout, MarlinQQQLayout)
    )


def _linear_int8_act_int4_weight_marlin_qqq_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import marlin_qqq_gemm
    from torchao.quantization.marlin_qqq import marlin_qqq_workspace

    assert isinstance(input_tensor, AffineQuantizedTensor)
    assert isinstance(weight_tensor, AffineQuantizedTensor)

    input = input_tensor.tensor_impl.int_data
    input_scale = input_tensor.tensor_impl.scale

    w_int4 = weight_tensor.tensor_impl.int_data
    s_group = weight_tensor.tensor_impl.s_group
    s_channel = weight_tensor.tensor_impl.s_channel
    original_shape = weight_tensor.tensor_impl.original_shape

    # Folds batch dimension into the first dimension
    input_2d = input.view(-1, input.shape[-1])
    input_scale = input_scale.view(1, -1)

    size_m = input_2d.shape[0]
    size_n = s_channel.shape[1]
    size_k = input_2d.shape[1]
    workspace_qqq = marlin_qqq_workspace(original_shape[1])

    out = marlin_qqq_gemm(
        input_2d,
        w_int4,
        input_scale,
        s_channel,
        s_group,
        workspace_qqq,
        size_m,
        size_n,
        size_k,
    )

    # Unfold the batch dimension
    out = out.reshape(input.shape[:-1] + (s_channel.shape[1],))

    if bias is not None:
        out += bias.to(out.dtype)
    return out


to_marlinqqq_quantized_intx = MarlinQQQTensor.from_hp_to_intx
