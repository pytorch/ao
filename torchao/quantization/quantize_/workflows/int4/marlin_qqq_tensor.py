# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import List

import torch

from torchao.quantization.quant_primitives import (
    _choose_qparams_and_quantize_affine_qqq,
)
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "MarlinQQQTensor",
]

aten = torch.ops.aten


class MarlinQQQTensor(TorchAOBaseTensor):
    """MarlinQQQ quantized tensor subclass which inherits AffineQuantizedTensor class.

    To see what happens during _choose_qparams_and_quantize_affine_qqq, quantization and dequantization for marlin qqq quantization,
    please checkout https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py
    and check the two quant primitive ops: _choose_qparams_and_quantize_affine_qqq and _dequantize_affine_qqq

    CHANGE THIS !!!!
    """
    tensor_data_names = ["qdata", "s_group", "s_channel"]
    tensor_attribute_names = ["original_shape", "group_size", "num_bits"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else qdata.layout
        )
        kwargs["dtype"] = qdata.dtype
        kwargs["requires_grad"] = False
        shape = qdata.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        s_group: torch.Tensor,
        s_channel: torch.Tensor,
        original_shape: torch.Size,
        group_size: int,
        num_bits: int,
    ):
        self.qdata = qdata
        self.s_group = s_group
        self.s_channel = s_channel
        self.original_shape = original_shape
        self.group_size = group_size
        self.num_bits = num_bits

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int]
    ):
        quant_min  = -8
        quant_max = 7

        """Converts a floating point tensor to a Marlin QQQ quantized tensor."""
        # input_float = _layout.pre_process(input_float) # i think this is a no-op
        nbits = int(math.log2(quant_max - quant_min + 1))
        group_size = max(block_size)
        wq, s_group, s_channel, _ = _choose_qparams_and_quantize_affine_qqq(
            w, nbits, group_size
        )

        ### packing ###

        from torchao.quantization.marlin_qqq import (
            const,
            pack_to_marlin_qqq,
        )

        # Linear layers are (in_features, out_features) but the wq that is reaching this point
        # is (out_features, in_features). We need to transpose it to match the expected shape in the marlin code.
        q_w = wq.t()
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
            qdata=marlin_qqq_q_w,
            s_group=marlin_qqq_s_group,
            s_channel=marlin_qqq_s_channel,
            original_shape=q_w.shape,
            group_size=group_size,
            num_bits=num_bits,
        )

implements = MarlinQQQTensor.implements

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale.is_contiguous(), "Expected scale to be contiguous"
    assert weight_tensor.zero_point.is_contiguous(), (
        "Expected zero_point to be contiguous"
    )

    from torchao.ops import marlin_qqq_gemm
    from torchao.quantization.marlin_qqq import marlin_qqq_workspace

    input = input_tensor.tensor_impl.qdata
    input_scale = input_tensor.tensor_impl.s_group

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


MarlinQQQTensor.__module__ = "torchao.quantization"

# Allow a model with MarlinQQQTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([MarlinQQQTensor])
