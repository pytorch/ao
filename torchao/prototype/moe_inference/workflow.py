# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import types
from typing import Callable

import torch

from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import (
    _float8_dynamic_activation_float8_weight_quantize_tensor,
    _linear_extra_repr,
)
from torchao.quantization.quantize_.common import (
    KernelPreference,
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
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
    xq = _choose_quant_func_and_quantize_tensor(x, w.act_quant_kwargs)
    xs = xq.scale.squeeze(-1)  # [*leading_dims, 1] - > [*leading_dims]
    ws = w.scale.squeeze(1)  # [B, 1, N] -> [B, N]
    yq = torch._scaled_grouped_mm(xq.qdata, w.qdata, xs, ws, offs=offs)
    return yq


# TODO(future PR): make the quantize_ API support quantization of parameters
# and remove this wrapper
def convert_to_float8_moe_inference(
    model: torch.nn.Module,
    # TODO(future PR): also support weight-only quant
    config: Float8DynamicActivationFloat8WeightConfig,
    param_filter_fn: Callable[[torch.nn.Parameter, str], bool],
) -> None:
    assert config.granularity == [PerRow(), PerRow()], (
        f"unsupported granularity {config.granularity}"
    )
    assert config.kernel_preference == KernelPreference.AUTO, (
        f"unsupported kernel_preference {config.kernel_preference}"
    )

    for mod_name, mod in model.named_modules():
        for param_name, old_param in mod.named_parameters():
            combined_name = f"{mod_name}.{param_name}"
            if not param_filter_fn(old_param, combined_name):
                continue
            new_param = _float8_dynamic_activation_float8_weight_quantize_tensor(
                old_param, config
            )
            setattr(mod, param_name, torch.nn.Parameter(new_param, requires_grad=False))
            mod.extra_repr = types.MethodType(_linear_extra_repr, mod)
