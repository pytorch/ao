# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch


class TwoStepQuantizer:
    """Base class for QAT quantizers that follow a two-step prepare + convert flow.

    Subclasses should implement:
    - ``prepare``: insert fake quantization into the model for QAT training
    - ``convert``: convert the fake-quantized model to a quantized model
    """

    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        raise NotImplementedError

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        raise NotImplementedError
