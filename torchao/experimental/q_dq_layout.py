# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.affine_quantized_tensor_ops import (
    register_aqt_quantized_linear_dispatch,
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


from torchao.dtypes.uintx.plain_layout import PlainAQTTensorImpl


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


register_aqt_quantized_linear_dispatch(
    _linear_check,
    _linear_impl,
)
