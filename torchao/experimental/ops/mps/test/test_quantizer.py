# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import copy
import itertools
import os
import sys

import torch
import unittest

from parameterized import parameterized
from torchao.experimental.quant_api import UIntxWeightOnlyLinearQuantizer
from torchao.experimental.quant_api import _quantize

libname = "libtorchao_ops_mps_linear_fp_act_xbit_weight_aten.dylib"
libpath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../cmake-out/lib/", libname)
)

try:
    for nbit in range(1, 8):
        getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
        getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
except AttributeError:
    try:
        torch.ops.load_library(libpath)
    except:
        raise RuntimeError(f"Failed to load library {libpath}")
    else:
        try:
            for nbit in range(1, 8):
                getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
                getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
        except AttributeError as e:
            raise e


class TestUIntxWeightOnlyLinearQuantizer(unittest.TestCase):
    BITWIDTHS = range(1, 8)
    GROUPSIZES = [32, 64, 128, 256]

    # Currently, the quantization code in quant_api.py only supports K values
    # multiple of group_size.
    # TODO(mcandales): Generalize the code in quant_api.py and add tests to
    # cover values of K not multiple of group_size.
    def _model_setup(self):
        group_size = 32
        k0 = 96
        k1 = 224
        k2 = 160
        n = 47
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, k2, bias=False),
            torch.nn.Linear(k2, n, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        return model, group_size, k0, n

    def _quantize_model(self, model, precision, nbit, group_size):
        quantizer = UIntxWeightOnlyLinearQuantizer(
            device="mps",
            precision=precision,
            bitwidth=nbit,
            groupsize=group_size,
        )
        quantized_model = copy.deepcopy(model)
        quantized_model = quantizer.quantize(quantized_model)
        return quantized_model

    @parameterized.expand(BITWIDTHS)
    def test_export(self, nbit):
        model, group_size, k0, n = self._model_setup()
        m = 3
        activations = torch.randn(m, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        exported = torch.export.export(quantized_model, (activations,))

        for node in exported.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(
                    str(node.target)
                    == f"torchao._linear_fp_act_{nbit}bit_weight.default"
                )

    @parameterized.expand(BITWIDTHS)
    def test_2d_output_device_and_shape(self, nbit):
        model, group_size, k0, n = self._model_setup()
        m = 3
        activations = torch.randn(m, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        result = quantized_model(activations)
        self.assertTrue(result.is_mps)
        self.assertTrue(result.shape == (m, n))

    @parameterized.expand(BITWIDTHS)
    def test_3d_output_device_and_shape(self, nbit):
        model, group_size, k0, n = self._model_setup()
        leading_shape = (3, 5)
        activations = torch.randn(*leading_shape, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        result = quantized_model(activations)
        self.assertTrue(result.is_mps)
        self.assertTrue(result.shape == (*leading_shape, n))

    @parameterized.expand(itertools.product(BITWIDTHS, GROUPSIZES))
    def test_valid_groupsizes(self, nbit, group_size):
        k0 = 3 * group_size
        k1 = 7 * group_size
        n = 47
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, n, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        m = 5
        activations = torch.randn(m, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        result = quantized_model(activations)
        self.assertTrue(result.is_mps)
        self.assertTrue(result.shape == (m, n))

    @parameterized.expand(BITWIDTHS)
    def test_invalid_groupsizes(self, nbit):
        group_size = 16
        k0 = 3 * group_size
        k1 = 7 * group_size
        n = 47
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, n, bias=False),
        ]
        model = torch.nn.Sequential(*layers)

        with self.assertRaises(ValueError):
            self._quantize_model(model, torch.float32, nbit, group_size)

    # TODO(mcandales): Consolidate with the reference impl in test_lowbit.py
    def _reference_linear_lowbit_quant_weights(self, A, W, group_size, S, Z):
        N = W.shape[0]
        K = W.shape[1]
        W = W.to(torch.float32)
        scales = S.t().unsqueeze(2).repeat(1, 1, group_size).view(N, -1)[:, :K]
        zeros = Z.t().unsqueeze(2).repeat(1, 1, group_size).view(N, -1)[:, :K]
        W = scales * W + zeros
        return torch.mm(A, W.t())

    @parameterized.expand(BITWIDTHS)
    def test_accuracy(self, nbit):
        group_size = 32
        m = 3
        n = 7
        k = 64
        with torch.no_grad():
            activations = torch.rand(m, k, dtype=torch.float32, device="mps")
            model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])
            quantized_model = self._quantize_model(
                model, torch.float32, nbit, group_size
            )
            result = quantized_model(activations)

            # Compute expected result
            weight_cpu = model[0].weight.data
            weight_qvals_cpu, weight_scales_cpu, weight_zeros_cpu = _quantize(
                weight_cpu, group_size, nbit, True, torch.uint8
            )
            weight_scales_cpu = weight_scales_cpu.t()
            weight_zeros_cpu = -weight_zeros_cpu.t() * weight_scales_cpu
            expected = self._reference_linear_lowbit_quant_weights(
                activations.cpu(),
                weight_qvals_cpu,
                group_size,
                weight_scales_cpu,
                weight_zeros_cpu,
            )

            # Compare results
            torch.testing.assert_close(result.cpu(), expected, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    unittest.main()
