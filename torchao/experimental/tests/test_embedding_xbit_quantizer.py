# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import glob
import subprocess
import os

import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from torchao.experimental.quant_api import (
    IntxWeightEmbeddingQuantizer,
    _IntxWeightQuantizedEmbeddingFallback,
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


class TestEmbeddingQuantizer(unittest.TestCase):
    def test_accuracy(self):
        group_size = 128
        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(*[torch.nn.Embedding(num_embeddings, embedding_dim)])
        indices = torch.randint(0, num_embeddings, (7,), dtype=torch.int32)

        for nbit in [1, 2, 3, 4, 5, 6, 7]:
            print(f"Testing nbit={nbit}")
            quantized_model = copy.deepcopy(model)
            quantizer = IntxWeightEmbeddingQuantizer(
                device="cpu",
                precision=torch.float32,
                bitwidth=nbit,
                groupsize=group_size,
            )
            quantized_model = quantizer.quantize(quantized_model)

            with torch.no_grad():
                result = quantized_model(indices)
                reference_impl = _IntxWeightQuantizedEmbeddingFallback(nbit)
                reference_impl.quantize_and_pack_weights(model[0].weight, group_size)
                expected_result = reference_impl(indices)
            self.assertTrue(torch.allclose(result, expected_result))

    def test_export_compile_aoti(self):
        nbit = 4
        group_size = 128
        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(*[torch.nn.Embedding(num_embeddings, embedding_dim)])
        indices = torch.randint(0, num_embeddings, (42,), dtype=torch.int32)


        print("Quantizing model")
        quantizer = IntxWeightEmbeddingQuantizer(
            device="cpu",
            precision=torch.float32,
            bitwidth=nbit,
            groupsize=group_size,
        )
        quantized_model = quantizer.quantize(model)

        print("Exporting quantized model")
        exported = torch.export.export(quantized_model, (indices,))

        print("Compiling quantized model")
        quantized_model_compiled = torch.compile(quantized_model)
        with torch.no_grad():
            quantized_model_compiled(indices)

        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Exporting quantized model with AOTI")
            torch._export.aot_compile(
                quantized_model,
                (indices,),
                options={"aot_inductor.output_path": f"{tmpdirname}/model.so"},
            )

            print("Running quantized model in AOTI")
            fn = torch._export.aot_load(f"{tmpdirname}/model.so", "cpu")
            fn(indices)

if __name__ == "__main__":
    unittest.main()
