# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import torch

from torchao.quantization.unified import TwoStepQuantizer
from torchao.quantization.quant_primitives import ZeroPointDomain


# TODO: change this to quant_primitives.Granularity
class QuantizationGranularity(Enum):
    PER_CHANNEL = "per_channel"
    PER_TOKEN = "per_token"
    PER_GROUP = "per_group"


@dataclass
class FakeQuantizeConfig:
    """
    Config for how to fake quantize weights or activations.

    args:
        bit_width: number of bits to simulate during fake quantization
        granularity: granularity of scales and zero points, one of:
            'per_token', 'per_channel', or 'per_group'
        group_size: size of each group for 'per_group' granularity
        symmetric: whether to use symmetric (default) or asymmetric quantization
        scale_precision: scale dtype (default torch.fp32)
        zero_point_precision: zero point dtype (default torch.int32)
        zero_point_domain: whether zero point is in integer (default) or float domain
        dynamic: whether to use dynamic (defualt) or static scale and zero points
        range_learning: whether to learn scale and zero points during training (coming soon)
    """
    bit_width: int
    granularity: Optional[QuantizationGranularity] = None
    group_size: Optional[int] = None
    symmetric: bool = True
    scale_precision: torch.dtype = torch.float32
    zero_point_precision: torch.dtype = torch.int32
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT
    dynamic: bool = True
    range_learning: bool = False

    def __post_init__(self):
        """
        Verify that `group_size` and `granularity` are consistent.
        """
        if self.group_size is None and self.granularity is None:
            raise ValueError("At least one of group_size or granularity must be set")
        if self.granularity == QuantizationGranularity.PER_GROUP and self.group_size is None:
            raise ValueError("Granularity is 'per_group' but no group_size was set")
        if self.granularity != QuantizationGranularity.PER_GROUP and self.group_size is not None:
            if self.granularity is None:
                self.granularity = QuantizationGranularity.PER_GROUP
            else:
                raise ValueError(
                    "Granularity is '%s' but group_size was set" % self.granularity.value
                )
        self._initialized = True

    def __setattr__(self, name: str, value: Any):
        """
        Support setting `granularity` by string and through `group_size`.
        """
        if name == "group_size" and getattr(self, "_initialized", False):
            super().__setattr__("granularity", QuantizationGranularity.PER_GROUP)
        if name == "granularity" and isinstance(value, str):
            value = QuantizationGranularity(value)
        super().__setattr__(name, value)


class ComposableQATQuantizer(TwoStepQuantizer):
    """
    Composable quantizer that users can use to apply multiple QAT quantizers easily.
    Quantizers will be applied in the order they are specified in the constructor.

    Note: the quantizers provided must apply to different modules in the model,
    e.g. nn.Linear and nn.Embedding, otherwise the behavior will be undefined.

    Example usage::

        my_quantizer = ComposableQATQuantizer([
            QATQuantizer1(),
            QATQuantizer2(),
            QATQuantizer3(),
        ])
        model = my_quantizer.prepare(model)
        train(model)
        model = my_quantizer.convert(model)
    """

    def __init__(self, quantizers: List[TwoStepQuantizer]):
        self.quantizers = quantizers

    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        for quantizer in self.quantizers:
            model = quantizer.prepare(model)
        return model

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        for quantizer in self.quantizers:
            model = quantizer.convert(model)
        return model
