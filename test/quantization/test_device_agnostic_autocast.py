# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4601.

Three call sites hardcoded CUDA autocast queries
(`torch.is_autocast_enabled()` with no argument, or explicitly
`torch.is_autocast_enabled("cuda")`, plus `torch.get_autocast_gpu_dtype()`),
so on a non-CUDA device with device autocast active they always saw
autocast as disabled and left the input in its original (usually fp32)
dtype instead of casting it to the active autocast dtype:

  - torchao/float8/float8_linear.py: Float8Linear.forward
  - torchao/prototype/quantized_training/int8_mixed_precision.py:
    the F.linear override for Int8MixedPrecisionTrainingLinearWeight
  - torchao/prototype/quantized_training/bitnet.py: the F.linear override
    for BitNetTrainingLinearWeight

Each test below runs entirely on CPU under `torch.autocast(device_type="cpu")`
and checks the dtype actually seen by the wrapped compute op, bypassing the
op itself (fp8/int8 matmul kernels are GPU-only) via a lightweight monkeypatch
so only the autocast-detection logic under test executes.
"""

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests


class TestDeviceAgnosticAutocast(TestCase):
    def test_float8_linear_respects_cpu_autocast(self):
        from torchao.float8 import float8_linear as m
        from torchao.float8.config import Float8LinearConfig

        captured = {}

        class FakeMatmul:
            @staticmethod
            def apply(input, weight_t, linear_mm_config, config):
                captured["dtype"] = input.dtype
                return input @ weight_t

        orig = m.matmul_with_hp_or_float8_args
        m.matmul_with_hp_or_float8_args = FakeMatmul
        try:
            lin = m.Float8Linear(
                8, 8, bias=False, config=Float8LinearConfig(emulate=True)
            )
            x = torch.randn(4, 8)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                lin(x)
        finally:
            m.matmul_with_hp_or_float8_args = orig

        self.assertEqual(captured["dtype"], torch.bfloat16)

    def test_int8_mixed_precision_training_respects_cpu_autocast(self):
        from torchao.prototype.quantized_training import int8_mixed_precision as m

        W = m.Int8MixedPrecisionTrainingLinearWeight
        fn = W._ATEN_OP_TABLE[W][F.linear]

        captured = {}

        class FakeFunction:
            @staticmethod
            def apply(input, weight, bias, config=None):
                captured["dtype"] = input.dtype
                return input

        orig = m._Int8MixedPrecisionTrainingLinearFunction
        m._Int8MixedPrecisionTrainingLinearFunction = FakeFunction
        try:
            weight = W(torch.randn(8, 8), m.Int8MixedPrecisionTrainingConfig())
            x = torch.randn(4, 8)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                fn(F.linear, (W,), (x, weight, None), {})
        finally:
            m._Int8MixedPrecisionTrainingLinearFunction = orig

        self.assertEqual(captured["dtype"], torch.bfloat16)

    def test_bitnet_training_respects_cpu_autocast(self):
        from torchao.prototype.quantized_training import bitnet as m

        W = m.BitNetTrainingLinearWeight
        fn = W._TORCH_FN_TABLE[W][F.linear]

        captured = {}

        class FakeFunction:
            @staticmethod
            def apply(input, weight, bias):
                captured["dtype"] = input.dtype
                return input

        orig = m._BitNetTrainingLinear
        m._BitNetTrainingLinear = FakeFunction
        try:
            weight = W(torch.randn(8, 8))
            x = torch.randn(4, 8)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                fn(F.linear, (W,), (x, weight, None), {})
        finally:
            m._BitNetTrainingLinear = orig

        self.assertEqual(captured["dtype"], torch.bfloat16)


if __name__ == "__main__":
    run_tests()
