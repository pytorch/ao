# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
)
from torchao.utils import DummyModule


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class AWQStep(str, Enum):
    PREPARE = "prepare"
    CONVERT = "convert"
    PREPARE_FOR_LOADING = "prepare_for_loading"


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


class AWQObserver(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        base_config: AOBaseConfig,
        scale_search_space_size: int = 20,
    ):
        """
        A custom observer for Activation aware Weight Quantization (AWQ)
        Note: this only applies to weight only quantization: https://github.com/pytorch/ao/issues/2388#issuecomment-3062863647

        Args:
            weight (torch.Tensor: The weight tensor to be observed.
            bias (Optional[torch.Tensor]): The bias tensor to be observed.
            config (AOBaseConfig): the configuration for quantize_, that we'll use to apply awq on top of
            scale_search_space_size (int): search space size for searching the best scale for weight and input activation
        """
        super().__init__()
        self.base_config = base_config
        self.weight = weight
        self.bias = bias
        self.inputs = []
        self.scale_options = scale_search_space_size
        self.device = self.weight.device
        if self.bias is not None:
            self.bias.to(self.device)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, output: torch.Tensor):
        self.inputs.append(input.to("cpu"))

    def calculate_qparams(self):
        assert self.inputs != None, (
            "calibrate observer first by running model on exemplar data"
        )
        for i in range(len(self.inputs)):
            self.inputs[i] = self.inputs[i].to(self.device)
            if self.bias is not None:
                self.bias = self.bias.to(self.device)

        acc = torch.cat(self.inputs, dim=-2)
        x_max = get_act_scale(acc)

        best_loss = float("inf")
        best_scales = None
        for i in range(self.scale_options):
            ratio = i * 1 / self.scale_options
            scales = x_max.pow(ratio).to(self.weight.dtype).clamp(min=1e-4).view(-1)
            if best_scales is None:
                best_scales = scales
            scales = scales / (scales.max() * scales.min()).sqrt()
            config_handler = _QUANTIZE_CONFIG_HANDLER[type(self.base_config)]
            dummy_mod = DummyModule(self.weight * scales)
            quant_mod = config_handler(dummy_mod, self.base_config)
            w = quant_mod.weight
            orig_out = F.linear(acc, self.weight, self.bias)
            q_out = F.linear(acc / scales, w, self.bias)
            loss = (orig_out - q_out).pow(2).mean().item()
            if loss < best_loss:
                best_scales = scales
                best_loss = loss
        return best_scales.detach()


class AWQObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: torch.nn.Module,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, act_obs: AWQObserver):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            False,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
