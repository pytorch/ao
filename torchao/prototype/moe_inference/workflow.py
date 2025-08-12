# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.quantization.quantize_.common import _choose_quant_func_and_quantize_tensor
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
    QuantizeTensorToFloat8Kwargs,
)

aten = torch.ops.aten


# set up an extra override for Float8Tensor weight and torch._grouped_mm
# TODO(future PR): move this to Float8Tensor once this logic is out of prototype
@Float8Tensor.implements([aten._grouped_mm.default])
def _(func, types, args, kwargs):
    # TODO(this PR): respect kernel preference
    x, w, offs = args
    assert isinstance(w, Float8Tensor)
    # quantize activation
    # TODO(future): also support weight-only quant
    xq = _choose_quant_func_and_quantize_tensor(x, w.act_quant_kwargs)
    xs = xq.scale.squeeze(-1)  # [*leading_dims, 1] - > [*leading_dims]
    ws = w.scale.squeeze(1)  # [B, 1, N] -> [B, N]
    yq = torch._scaled_grouped_mm(xq.qdata, w.qdata, xs, ws, offs=offs)
    return yq


# TODO(before land): use the quantize_ API instead of hardcoding
def convert_to_float8_moe_inference(model: torch.nn.Module, weight_name: str) -> None:
    # replace weight with tensor subclass wrapper
    old_weight = getattr(model, weight_name)
    # print("old", old_weight)

    # dynamic quant - specify activation quant kwargs
    act_quant_kwargs = QuantizeTensorToFloat8Kwargs()

    with torch.no_grad():
        new_weight = Float8Tensor.to_float8(
            old_weight, act_quant_kwargs=act_quant_kwargs
        )
    # print("new", new_weight)
    setattr(model, weight_name, torch.nn.Parameter(new_weight))
    # print('old', old_weight.shape, 'new', new_weight.shape)
    # print(model)
