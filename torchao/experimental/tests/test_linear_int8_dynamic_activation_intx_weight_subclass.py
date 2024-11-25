# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import glob
import os
import subprocess

import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from torchao.experimental.quant_api import int8_dynamic_activation_intx_weight
from torchao.quantization.quant_api import quantize_

from torchao.utils import unwrap_tensor_subclass
from torchao.experimental.quant_api import (
    _Int8DynActIntxWeightQuantizedLinearFallback,
)

def cmake_build_torchao_ops(temp_build_dir):
    from distutils.sysconfig import get_python_lib

    print("Building torchao ops for ATen target")
    cmake_prefix_path = get_python_lib()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    subprocess.run(
        [
            "cmake",
            "-DCMAKE_PREFIX_PATH=" + cmake_prefix_path,
            "-DCMAKE_INSTALL_PREFIX=" + temp_build_dir.name,
            "-S " + dir_path + "/../",
            "-B " + temp_build_dir.name,
        ]
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            temp_build_dir.name,
            "-j 16",
            "--target install",
            "--config Release",
        ]
    )


temp_build_dir = tempfile.TemporaryDirectory()
cmake_build_torchao_ops(temp_build_dir)
libs = glob.glob(f"{temp_build_dir.name}/lib/libtorchao_ops_aten.*")
libs = list(filter(lambda l: (l.endswith("so") or l.endswith("dylib")), libs))
assert len(libs) == 1
torch.ops.load_library(libs[0])


class TestInt8DynamicActivationIntxWeight(unittest.TestCase):
    def test_accuracy(self):
        group_size = 128
        m = 1
        n = 1071
        k = 4096
        activations = torch.randn(m, k, dtype=torch.float32)
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for nbit in [1, 2, 3, 4, 5, 6, 7, 8]:
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

                    #TODO: remove expected_result2 checks when we deprecate non-subclass API
                    reference_impl = _Int8DynActIntxWeightQuantizedLinearFallback()
                    reference_impl.quantize_and_pack_weights(
                        model[0].weight, nbit, group_size, has_weight_zeros
                    )
                    expected_result2 = reference_impl(activations)

                num_mismatch_at_low_tol = 0
                num_mismatch_at_low_tol2 = 0
                num_total = result.reshape(-1).shape[0]
                for i in range(num_total):
                    actual_val = result.reshape(-1)[i]
                    expected_val = expected_result.reshape(-1)[i]
                    expected_val2 = expected_result2.reshape(-1)[i]
                    self.assertTrue(torch.allclose(actual_val, expected_val, atol=1e-6))
                    if not torch.allclose(actual_val, expected_val):
                        num_mismatch_at_low_tol += 1
                    
                    self.assertTrue(torch.allclose(expected_val, expected_val2, atol=1e-2, rtol=1e-1))
                    if not torch.allclose(expected_val, expected_val2):
                        num_mismatch_at_low_tol2 += 1

                # Assert at most 5% of entries are not close at a low tolerance
                self.assertTrue(num_mismatch_at_low_tol / num_total <= 0.05)
                self.assertTrue(num_mismatch_at_low_tol2 / num_total <= 0.01)

    def test_export_compile_aoti(self):
        group_size = 32
        m = 3
        k0 = 512
        k1 = 256
        k2 = 128
        k3 = 1024
        nbit = 4
        has_weight_zeros = True
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
        
        print("Compiling quantized model")
        compiled = torch.compile(unwrapped_model)
        with torch.no_grad():
            compiled(activations)

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
