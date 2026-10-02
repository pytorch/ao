# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4600.

gptq_quantize() previously had two CUDA-only lines:
  - `assert device.type == "cuda", "GPTQ only supports CUDA currently"`
  - `torch.cuda.synchronize()`
even though the GPTQ algorithm itself (torch.diag, torch.linalg, matmul,
indexing) is pure PyTorch and device-agnostic. This test runs it on CPU,
which used to fail with an AssertionError (or, if that assert were removed
by hand, a `RuntimeError: No CUDA GPUs are available` from the
unconditional torch.cuda.synchronize()).
"""

import pytest
import torch
import torch.nn.functional as F

from torchao.utils import torch_version_at_least

pytestmark = pytest.mark.skipif(
    not torch_version_at_least("2.11.0"),
    reason="GPTQ prototype requires PyTorch 2.11+",
)

from torchao.prototype.gptq import GPTQConfig, gptq_quantize
from torchao.quantization import Int8Tensor, Int8WeightOnlyConfig
from torchao.quantization.granularity import PerRow


def test_gptq_quantize_runs_on_cpu():
    torch.manual_seed(42)

    out_features = 32
    in_features = 64
    weight = torch.randn(out_features, in_features, dtype=torch.float32)

    # Synthetic positive-definite Hessian.
    A = torch.randn(in_features, in_features, dtype=torch.float32)
    H = A.t() @ A + torch.eye(in_features) * 0.1

    config = GPTQConfig(
        step="convert",
        base_config=Int8WeightOnlyConfig(granularity=PerRow()),
    )

    # This used to raise:
    #   AssertionError: GPTQ only supports CUDA currently
    quantized_weight = gptq_quantize(H, weight, config)

    assert isinstance(quantized_weight, Int8Tensor)
    assert quantized_weight.shape == weight.shape

    dequantized = F.linear(
        torch.eye(in_features, dtype=torch.float32),
        quantized_weight,
        None,
    ).t()
    assert dequantized.shape == weight.shape
