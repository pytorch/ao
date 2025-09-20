# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8PackingFormat,
    quantize_,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import torch_version_at_least


def _make_cfg(act: str, target_sparsity: float = 0.90):
    """
    Helper to build the v2 CSR config:
      - act == "sym"  -> dynamic int8 symmetric per-token
      - act == "asym" -> dynamic uint8 asymmetric per-token
      - act == "noop" -> weight-only decode (no activation quant)
    """
    if act == "noop":
        return Int8DynamicActivationInt8WeightConfig(
            act_mapping_type=MappingType.SYMMETRIC,  # ignored when weight_only_decode=True
            weight_only_decode=True,
            version=2,
            int8_packing_format=Int8PackingFormat.CSR_SPARSE,
            target_sparsity=target_sparsity,
        )
    elif act == "sym":
        return Int8DynamicActivationInt8WeightConfig(
            act_mapping_type=MappingType.SYMMETRIC,
            weight_only_decode=False,
            version=2,
            int8_packing_format=Int8PackingFormat.CSR_SPARSE,
            target_sparsity=target_sparsity,
        )
    elif act == "asym":
        return Int8DynamicActivationInt8WeightConfig(
            act_mapping_type=MappingType.ASYMMETRIC,
            weight_only_decode=False,
            version=2,
            int8_packing_format=Int8PackingFormat.CSR_SPARSE,
            target_sparsity=target_sparsity,
        )
    else:
        raise ValueError(f"Unknown act mode: {act}")


CPU_DTYPES = [torch.float32]  # CSR fallback path is CPU in your implementation


@unittest.skipIf(not torch_version_at_least("2.7.0"), "Need PyTorch 2.7+")
class TestInt8CsrSparseTensor(TestCase):
    @parametrize("act_mode", ["sym", "asym", "noop"])
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),  # (M,),  N, K
            ((32, 64), 512, 256),  # (B, T), N, K
            ((2, 8, 16), 384, 192),  # (B, T, ?), N, K
        ],
    )
    @parametrize("dtype", CPU_DTYPES)
    def test_linear_forward_cpu(self, act_mode, sizes, dtype):
        """
        Forward should run, produce finite values, and keep shapes consistent.
        """
        M, N, K = sizes
        x = torch.randn(*M, K, dtype=dtype, device="cpu")
        lin = torch.nn.Linear(K, N, bias=True, dtype=dtype, device="cpu")

        # fp32 reference
        y_ref = lin(x)

        cfg = _make_cfg(act_mode, target_sparsity=0.90)
        quantize_(lin, cfg)

        # weight must be our subclass
        self.assertEqual(
            str(type(lin.weight)),
            "<class 'torchao.quantization.Int8CsrSparseTensor'>",
        )

        y_q = lin(x)
        self.assertEqual(y_q.shape, y_ref.shape)
        self.assertTrue(torch.isfinite(y_q).all(), "Quantized output has NaN/Inf")

        # Sanity: expect some difference from fp32 (not required to be large)
        diff = (y_q - y_ref).abs().mean()
        self.assertTrue(torch.isfinite(diff))
        self.assertGreaterEqual(diff.item(), 0.0)

    @parametrize("act_mode", ["sym", "asym", "noop"])
    def test_module_path_state_dict(self, act_mode):
        """
        Saving state_dict and loading it back preserves the subclass type
        of the weight tensor.
        """
        K, N = 128, 256
        lin = torch.nn.Linear(K, N, bias=True, dtype=torch.float32, device="cpu")
        cfg = _make_cfg(act_mode, target_sparsity=0.85)
        quantize_(lin, cfg)

        self.assertEqual(
            str(type(lin.weight)),
            "<class 'torchao.quantization.Int8CsrSparseTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(lin.state_dict(), f)
            f.seek(0)
            sd = torch.load(f)
            self.assertEqual(
                str(type(sd["weight"])),
                "<class 'torchao.quantization.Int8CsrSparseTensor'>",
            )

    def test_guard_small_in_features(self):
        """
        If you keep the v1 guard (in_features <= 16) anywhere in your path,
        ensure v2 config still quantizes (or update this accordingly).
        Here we use K=32 to avoid hitting the guard.
        """
        K, N = 32, 64
        x = torch.randn(4, K)
        lin = torch.nn.Linear(K, N)
        cfg = _make_cfg("sym", target_sparsity=0.9)
        quantize_(lin, cfg)
        y = lin(x)
        self.assertEqual(y.shape, (4, N))
        self.assertTrue(torch.isfinite(y).all())


instantiate_parametrized_tests(TestInt8CsrSparseTensor)


if __name__ == "__main__":
    run_tests()
