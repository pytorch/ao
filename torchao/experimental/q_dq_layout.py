# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.affine_quantized_tensor_ops import (
    register_aqt_quantized_linear_dispatch,
)
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


from torchao.dtypes.utils import PlainLayout


class QDQLayout(PlainLayout):
    pass


from torchao._executorch_ops import (
    _quantized_decomposed_dequantize_per_channel_group_wrapper,
)
from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl
from torchao.quantization.utils import per_token_dynamic_quant


@register_layout(QDQLayout)
class _Impl(PlainAQTTensorImpl):
    pass


def _linear_check(input_tensor, weight_tensor, bias):
    layout = weight_tensor.tensor_impl.get_layout()
    return isinstance(layout, QDQLayout)


def _linear_impl(input_tensor, weight_tensor, bias):
    if isinstance(input_tensor, AffineQuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight_tensor, AffineQuantizedTensor):
        weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

    # assert isinstance(input_tensor, )
    # if isinstance(input_tensor, AffineQuantizedTensor):


    #         input_tensor = input_tensor.dequantize()
    #     if isinstance(weight_tensor, AffineQuantizedTensor):
    #         weight_tensor = weight_tensor.dequantize()
    #     return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

    # x = per_token_dynamic_quant(input_tensor)

    # w_int_data = weight_tensor.tensor_impl.int_data
    # w_scale = weight_tensor.tensor_impl.scale
    # w_zero_point = weight_tensor.tensor_impl.zero_point
    # assert len(weight_tensor.block_size) == 2
    # assert weight_tensor.block_size[0] == 1
    # group_size = weight_tensor.block_size[1]

    # w_dq = _quantized_decomposed_dequantize_per_channel_group_wrapper(
    #     w_int_data,
    #     w_scale,
    #     w_zero_point,
    #     weight_tensor.quant_min,
    #     weight_tensor.quant_max,
    #     torch.int8,
    #     group_size,
    #     torch.float32,
    # )

    # return torch.nn.functional.linear(x, w_dq, bias)


register_aqt_quantized_linear_dispatch(
    _linear_check,
    _linear_impl,
)
