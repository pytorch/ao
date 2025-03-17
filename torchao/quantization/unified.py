# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Any

import torch

"""
The vast majority of quantization algorithms follow one of two patterns
1. Single quantize call to create a quantized model with quantized state_dict
2. Flow that needs calibration or training

This file defines the API for both patterns
"""


# API 1, single quantize call to create a quantized model with quantized state_dict
class Quantizer(ABC):
    @abstractmethod
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        pass


# API 2, flow that needs calibration or training
class TwoStepQuantizer:
    @abstractmethod
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        pass

    @abstractmethod
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        pass
