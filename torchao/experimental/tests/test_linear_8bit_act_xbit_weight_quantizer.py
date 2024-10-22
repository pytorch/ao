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
from torchao.experimental.quant_api import (
    _Int8DynActIntxWeightQuantizedLinearFallback,
    Int8DynActIntxWeightLinearQuantizer,
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


class TestInt8DynActIntxWeightQuantizer(unittest.TestCase):
    def test_accuracy(self):
        group_size = 128
        m = 1
        n = 1071
        k = 4096
        activations = torch.randn(2, 3, m, k, dtype=torch.float32)
        model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

        for nbit in [1, 2, 3, 4, 5, 6, 7]:
            for has_weight_zeros in [True, False]:
                print(f"Testing nbit={nbit}, has_weight_zeros={has_weight_zeros}")
                quantized_model = copy.deepcopy(model)
                quantizer = Int8DynActIntxWeightLinearQuantizer(
                    device="cpu",
                    precision=torch.float32,
                    bitwidth=nbit,
                    groupsize=group_size,
                    has_weight_zeros=has_weight_zeros,
                )
                quantized_model = quantizer.quantize(quantized_model)

                with torch.no_grad():
                    result = quantized_model(activations)
                    reference_impl = _Int8DynActIntxWeightQuantizedLinearFallback()
                    reference_impl.quantize_and_pack_weights(
                        model[0].weight, nbit, group_size, has_weight_zeros
                    )
                    expected_result = reference_impl(activations)

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

        activations = torch.randn(m, k0, dtype=torch.float32)

        print("Quantizing model")
        quantizer = Int8DynActIntxWeightLinearQuantizer(
            device="cpu",
            precision=torch.float32,
            bitwidth=nbit,
            groupsize=group_size,
            has_weight_zeros=has_weight_zeros,
        )
        quantized_model = quantizer.quantize(model)

        print("Exporting quantized model")
        exported = torch.export.export(quantized_model, (activations,))

        print("Compiling quantized model")
        quantized_model_compiled = torch.compile(quantized_model)
        with torch.no_grad():
            quantized_model_compiled(activations)

        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Exporting quantized model with AOTI")
            torch._export.aot_compile(
                quantized_model,
                (activations,),
                options={"aot_inductor.output_path": f"{tmpdirname}/model.so"},
            )

            print("Running quantized model in AOTI")
            fn = torch._export.aot_load(f"{tmpdirname}/model.so", "cpu")
            fn(activations)


if __name__ == "__main__":
    unittest.main()
