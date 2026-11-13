# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor import _dispatch, _ops

import torchao.prototype.moe_training.tensor as tensor_mod
from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
from torchao.prototype.moe_training.tensor import (
    MXFP8TrainingWeightWrapperTensor,
)
from torchao.quantization import quantize_


def _remove_mxfp8_dtensor_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(_ops, "scaled_mm_single_dim_strategy", raising=False)
    monkeypatch.delattr(_dispatch, "is_pinned_handler", raising=False)


def test_mxfp8_non_dtensor_conversion_does_not_require_torch_2_12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _remove_mxfp8_dtensor_support(monkeypatch)

    config = MXFP8TrainingOpConfig()
    model = nn.Linear(4, 4)
    quantize_(model, config)

    assert isinstance(model.weight, MXFP8TrainingWeightWrapperTensor)


def test_mxfp8_dtensor_requires_torch_2_12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _remove_mxfp8_dtensor_support(monkeypatch)
    monkeypatch.setattr(tensor_mod, "DTensor", torch.Tensor)

    with pytest.raises(
        RuntimeError,
        match="MXFP8 training with DTensor requires PyTorch 2.12 or later",
    ):
        MXFP8TrainingWeightWrapperTensor(torch.randn(4, 4), MXFP8TrainingOpConfig())
