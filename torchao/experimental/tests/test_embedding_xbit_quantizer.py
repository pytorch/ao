# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
from parameterized import param, parameterized
from torch.testing import FileCheck

from torchao.dtypes import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
)
from torchao.experimental.quant_api import (
    EmbeddingQuantizer,
    SharedEmbeddingQuantizer,
)
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.qat import (
    FakeQuantizeConfig,
    FromIntXQuantizationAwareTrainingConfig,
    Int4WeightOnlyEmbeddingQATQuantizer,
    IntXQuantizationAwareTrainingConfig,
)
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    quantize_,
)


class TestEmbeddingQuantizer(unittest.TestCase):
    def test_accuracy(self):
        granularity = PerGroup(128)
        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(
            *[torch.nn.Embedding(num_embeddings, embedding_dim)]
        )
        indices = torch.randint(0, num_embeddings, (7,), dtype=torch.int32)

        for weight_dtype, granularity, mapping_type in zip(
            list(getattr(torch, f"int{x}") for x in range(1, 9)),
            [PerGroup(128), PerAxis(0)],
            [MappingType.ASYMMETRIC, MappingType.SYMMETRIC],
        ):
            print(
                f"Testing weight_dtype={weight_dtype}, granularity={granularity}, mapping_type={mapping_type}"
            )
            quantized_model = copy.deepcopy(model)
            quantizer = EmbeddingQuantizer(
                weight_dtype=weight_dtype,
                granularity=granularity,
                mapping_type=mapping_type,
                use_fallback=False,
            )
            quantized_model = quantizer.quantize(quantized_model)

            with torch.no_grad():
                reference_quantizer = EmbeddingQuantizer(
                    weight_dtype=weight_dtype,
                    granularity=granularity,
                    mapping_type=mapping_type,
                    use_fallback=True,
                )
                reference_model = copy.deepcopy(model)
                reference_model = reference_quantizer.quantize(reference_model)
                result = quantized_model(indices)
                expected_result = reference_model(indices)
            self.assertTrue(torch.allclose(result, expected_result))

    def test_export_compile_aoti(self):
        weight_dtype = torch.int4
        granularity = PerAxis(0)
        weight_mapping_type = MappingType.ASYMMETRIC
        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(
            *[torch.nn.Embedding(num_embeddings, embedding_dim)]
        )
        indices = torch.randint(0, num_embeddings, (42,), dtype=torch.int32)

        print("Quantizing model")
        quantizer = EmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=weight_mapping_type,
            use_fallback=False,
        )
        quantized_model = quantizer.quantize(model)
        eager_results = model(indices)

        print("Exporting quantized model")
        with torch.no_grad():
            exported_model = torch.export.export(
                quantized_model, (indices,), strict=True
            )
            exported_results = exported_model.module()(indices)
            self.assertTrue(torch.allclose(eager_results, exported_results))

        print("Compiling quantized model")
        quantized_model_compiled = torch.compile(quantized_model)
        with torch.no_grad():
            quantized_model_compiled(indices)
            compiled_results = quantized_model_compiled(indices)
            self.assertTrue(torch.allclose(eager_results, compiled_results))

        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Exporting quantized model with AOTI")
            package_path = f"{tmpdirname}/model.pt2"
            torch._inductor.aoti_compile_and_package(
                exported_model, package_path=package_path
            )
            fn = torch._inductor.aoti_load_package(package_path)
            aoti_results = fn(indices)
            self.assertTrue(torch.allclose(eager_results, aoti_results))

    def test_shared_embedding(self):
        weight_dtype = torch.int4
        weight_mapping_type = MappingType.ASYMMETRIC
        embedding_dim = 4096
        num_embeddings = 131
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        unembedding = torch.nn.Linear(embedding_dim, num_embeddings)
        unembedding.weight = copy.deepcopy(embedding.weight)
        model = torch.nn.Sequential(
            *[
                embedding,
                torch.nn.Linear(embedding_dim, embedding_dim),
                unembedding,
            ]
        )
        indices = torch.randint(0, num_embeddings, (42,), dtype=torch.int32)

        # Reference implementation quantizes the embedding and unembedding
        # layers separately
        quantized_model_reference = copy.deepcopy(model)
        EmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=PerAxis(0),
            mapping_type=weight_mapping_type,
        ).quantize(quantized_model_reference)
        quantize_(
            quantized_model_reference,
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=weight_dtype,
                weight_granularity=PerAxis(0),
                weight_mapping_type=weight_mapping_type,
                layout=PackedLinearInt8DynamicActivationIntxWeightLayout(
                    target="universal"
                ),
            ),
            filter_fn=lambda m, fqn: fqn == "2",
        )

        # Do shared embedding quantization
        quantized_model = copy.deepcopy(model)
        SharedEmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=PerAxis(0),
            mapping_type=weight_mapping_type,
        ).quantize(quantized_model)

        # Check results are same and weights share the same id
        with torch.no_grad():
            result = quantized_model(indices)
            expected_result = quantized_model_reference(indices)
        self.assertTrue(torch.allclose(result, expected_result))
        self.assertTrue(
            id(quantized_model[0].unembedding_packed_weights)
            == id(quantized_model[2].packed_weight)
        )

        # Test export
        exported_program = torch.export.export(quantized_model, (indices,))
        exported_result = exported_program.module()(indices)
        self.assertTrue(torch.allclose(result, exported_result))

        # Check the shared_embedding and linear ops use the same lifted weight
        weight = "b_getattr_l__fn_____0___unembedding_packed_weights"
        expected_lines = [
            f"torch.ops.torchao._shared_embedding_4bit.default({weight}, 4096, 131, 4096, reshape)",
            f"torch.ops.torchao._linear_8bit_act_4bit_weight.default(linear, {weight}, 4096, 131, 4096)",
        ]
        for line in expected_lines:
            FileCheck().check_count(line, 1, exactly=True).run(
                exported_program.graph_module.code
            )

    @parameterized.expand(
        [
            param(
                weight_dtype=weight_dtype,
                granularity=granularity,
                mapping_type=mapping_type,
                model_dtype=model_dtype,
            )
            for weight_dtype in [getattr(torch, f"int{x}") for x in range(1, 9)]
            for granularity in [PerGroup(32), PerGroup(128), PerAxis(0)]
            for mapping_type in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
            for model_dtype in [torch.float32]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_identical_to_IntxWeightOnlyConfig(
        self, weight_dtype, granularity, mapping_type, model_dtype
    ):
        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(
            *[torch.nn.Embedding(num_embeddings, embedding_dim)]
        )
        model = model.to(model_dtype)
        indices = torch.randint(0, num_embeddings, (7,), dtype=torch.int32)

        quantized_model = copy.deepcopy(model)
        quantizer = EmbeddingQuantizer(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=mapping_type,
        )
        quantized_model = quantizer.quantize(quantized_model)
        actual_result = quantized_model(indices)

        reference_model = copy.deepcopy(model)
        quantize_(
            reference_model,
            IntxWeightOnlyConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                mapping_type=mapping_type,
                scale_dtype=None,
            ),
            lambda m, fqn: isinstance(m, torch.nn.Embedding),
        )
        expected_result = reference_model(indices)
        self.assertTrue(torch.allclose(actual_result, expected_result))

    @parameterized.expand(
        [
            param(
                weight_dtype=weight_dtype,
                granularity=granularity,
                mapping_type=mapping_type,
                scale_dtype=scale_dtype,
                model_dtype=model_dtype,
            )
            for weight_dtype in [getattr(torch, f"int{x}") for x in range(1, 9)]
            for granularity in [PerGroup(32), PerGroup(128), PerAxis(0)]
            for mapping_type in [MappingType.SYMMETRIC, MappingType.ASYMMETRIC]
            for scale_dtype in [torch.float32, torch.bfloat16, torch.float16]
            for model_dtype in [torch.float32, torch.bfloat16]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_identical_to_IntXQuantizationAwareTrainingConfig(
        self, weight_dtype, granularity, mapping_type, scale_dtype, model_dtype
    ):
        # ASYMMETRIC in QAT is very different that PTQ configs
        if mapping_type == MappingType.ASYMMETRIC:
            return

        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(
            *[torch.nn.Embedding(num_embeddings, embedding_dim)]
        )
        model = model.to(model_dtype)
        indices = torch.randint(0, num_embeddings, (7,), dtype=torch.int32)

        is_symmetric = mapping_type == MappingType.SYMMETRIC
        group_size = (
            granularity.group_size
            if isinstance(granularity, PerGroup)
            else embedding_dim
        )

        embedding_filter = lambda m, fqn: isinstance(m, torch.nn.Embedding)
        weight_config = FakeQuantizeConfig(
            weight_dtype,
            group_size=group_size,
            is_symmetric=is_symmetric,
            scale_precision=scale_dtype,
        )
        quantize_(
            model,
            IntXQuantizationAwareTrainingConfig(weight_config=weight_config),
            embedding_filter,
        )
        expected_out = model(indices)

        quantize_(model, FromIntXQuantizationAwareTrainingConfig(), embedding_filter)
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=weight_dtype,
                granularity=granularity,
                mapping_type=mapping_type,
                scale_dtype=scale_dtype,
            ),
            embedding_filter,
        )
        actual_out = model(indices)
        self.assertTrue(torch.allclose(expected_out, actual_out))

    @parameterized.expand(
        [
            param(
                granularity=granularity,
                scale_dtype=scale_dtype,
                model_dtype=model_dtype,
            )
            for granularity in [PerGroup(32), PerGroup(128), PerAxis(0)]
            for scale_dtype in [torch.float32, torch.bfloat16, torch.float16]
            for model_dtype in [torch.float32, torch.bfloat16]
        ],
        name_func=lambda f, _, params: f.__name__ + f"_{params.kwargs}",
    )
    def test_identical_to_Int4WeightOnlyEmbeddingQATQuantizer(
        self, granularity, scale_dtype, model_dtype
    ):
        embedding_dim = 4096
        num_embeddings = 131
        model = torch.nn.Sequential(
            *[torch.nn.Embedding(num_embeddings, embedding_dim)]
        )
        model = model.to(model_dtype)
        indices = torch.randint(0, num_embeddings, (7,), dtype=torch.int32)

        group_size = (
            granularity.group_size
            if isinstance(granularity, PerGroup)
            else embedding_dim
        )

        embedding_filter = lambda m, fqn: isinstance(m, torch.nn.Embedding)

        qat_quantizer = Int4WeightOnlyEmbeddingQATQuantizer(
            group_size=group_size,
            scale_precision=scale_dtype,
            zero_point_precision=torch.int32,
        )
        model = qat_quantizer.prepare(model)
        prepared_model_copy = copy.deepcopy(model)
        expected_out = model(indices)

        # Convert model method 1
        quantize_(model, FromIntXQuantizationAwareTrainingConfig(), embedding_filter)
        quantize_(
            model,
            IntxWeightOnlyConfig(
                weight_dtype=torch.int4,
                granularity=granularity,
                mapping_type=MappingType.SYMMETRIC,
                scale_dtype=scale_dtype,
            ),
            embedding_filter,
        )
        actual_out1 = model(indices)
        self.assertTrue(torch.allclose(expected_out, actual_out1))

        # Convert model method 2
        qat_quantizer.convert(prepared_model_copy)
        actual_out2 = prepared_model_copy(indices)
        self.assertTrue(torch.allclose(expected_out, actual_out2))


if __name__ == "__main__":
    unittest.main()
