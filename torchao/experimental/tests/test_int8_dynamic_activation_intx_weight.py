# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import glob
import os

import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quant_api import int8_dynamic_activation_intx_weight
from torchao.quantization.quant_api import quantize_

from torchao.utils import unwrap_tensor_subclass

libs = glob.glob("/tmp/cmake-out/torchao/lib/liblinear_a8wxdq_ATEN.*")
libs = list(filter(lambda l: (l.endswith("so") or l.endswith("dylib")), libs))
if len(libs) == 0:
    logger.warning(
        "Could not find library; please run `sh build_torchao_op.sh ATEN` to build the library.  A slow fallback kernel will be used instaed."
    )
else:
    torch.ops.load_library(libs[0])


class TestInt8DynamicActivationIntxWeight(unittest.TestCase):
    def test_accuracy(self):
        group_size = 128
        m = 1
        n = 1071
        k = 4096
        activations = torch.randn(m, k, dtype=torch.float32)
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for nbit in [1, 2, 3, 4, 5, 6, 7]:
            for has_weight_zeros in [True, False]:
                print(f"Testing nbit={nbit}, has_weight_zeros={has_weight_zeros}")
                quantized_model = copy.deepcopy(model)
                quantize_(
                    quantized_model,
                    int8_dynamic_activation_intx_weight(
                        group_size=group_size,
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                    ),
                )

                quantized_model_reference = copy.deepcopy(model)
                quantize_(
                    quantized_model_reference,
                    int8_dynamic_activation_intx_weight(
                        group_size=group_size,
                        nbit=nbit,
                        has_weight_zeros=has_weight_zeros,
                        target="fallback",
                    ),
                )

                with torch.no_grad():
                    result = quantized_model(activations)
                    expected_result = quantized_model_reference(activations)

                num_mismatch_at_low_tol = 0
                num_total = result.reshape(-1).shape[0]
                for i in range(num_total):
                    actual_val = result.reshape(-1)[i]
                    expected_val = expected_result.reshape(-1)[i]
                    self.assertTrue(torch.allclose(actual_val, expected_val, atol=1e-6))
                    if not torch.allclose(actual_val, expected_val):
                        num_mismatch_at_low_tol += 1

                # Assert at most 5% of entries are not close at a low tolerance
                self.assertTrue(num_mismatch_at_low_tol / num_total <= 0.05)

    def test_export_compile_aoti(self):
        group_size = 32
        m = 3
        k0 = 512
        k1 = 256
        k2 = 128
        k3 = 1024
        nbit = 4
        has_weight_zeros = False
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, k2, bias=False),
            torch.nn.Linear(k2, k3, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        activations = torch.randn(2, 1, m, k0, dtype=torch.float32)

        print("Quantizing model")
        quantize_(
            model,
            int8_dynamic_activation_intx_weight(
                group_size=group_size,
                nbit=nbit,
                has_weight_zeros=has_weight_zeros,
                target="native",
            ),
        )

        unwrapped_model = copy.deepcopy(model)
        unwrap_tensor_subclass(model)

        print("Exporting quantized model")
        exported = torch.export.export(model, (activations,))

        # @nocommit: compile does not work because AffineQuantizedTensor calls get_plain
        # in its repr
        # print("Compiling quantized model")
        # compiled = torch.compile(unwrapped_model)
        # with torch.no_grad():
        #     compiled(activations)

        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Exporting quantized model with AOTI")
            torch._export.aot_compile(
                model,
                (activations,),
                options={"aot_inductor.output_path": f"{tmpdirname}/model.so"},
            )

            print("Running quantized model in AOTI")
            fn = torch._export.aot_load(f"{tmpdirname}/model.so", "cpu")
            fn(activations)


if __name__ == "__main__":
    unittest.main()
