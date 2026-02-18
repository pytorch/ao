# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.quantization import PerAxis, PerGroup, PerTensor
from torchao.quantization.pt2e._affine_quantization import (
    AffineQuantizedMinMaxObserver,
    AffineQuantizedMovingAverageMinMaxObserver,
)
from torchao.quantization.pt2e.fake_quantize import (
    AffineFakeQuantize,
    default_affine_fake_quant,
    default_affine_per_axis_fake_quant,
    default_groupwise_fake_quant,
)
from torchao.quantization.pt2e.observer import MappingType
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer import (
    ComposableQuantizer,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)
from torchao.quantization.quant_primitives import (
    _fake_quantize_affine,
)
from torchao.testing.pt2e._xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


class AffineFakeQuantizeInitializationTest(unittest.TestCase):
    """Test initialization of AffineFakeQuantize with different configurations."""

    def test_init_with_affine_quantized_min_max_observer(self):
        """Test initialization with AffineQuantizedMinMaxObserver."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )
        self.assertIsInstance(fq.activation_post_process, AffineQuantizedMinMaxObserver)
        self.assertEqual(fq.target_dtype, torch.uint8)
        self.assertEqual(fq.mapping_type, MappingType.ASYMMETRIC)
        self.assertIsInstance(fq.granularity, PerTensor)
        self.assertFalse(fq.is_dynamic)

    def test_init_with_moving_average_observer(self):
        """Test initialization with AffineQuantizedMovingAverageMinMaxObserver."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMovingAverageMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerAxis(0),
        )
        self.assertIsInstance(
            fq.activation_post_process, AffineQuantizedMovingAverageMinMaxObserver
        )
        self.assertEqual(fq.target_dtype, torch.int8)
        self.assertEqual(fq.mapping_type, MappingType.SYMMETRIC)
        self.assertIsInstance(fq.granularity, PerAxis)

    def test_init_with_per_tensor_granularity(self):
        """Test initialization with PerTensor granularity."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )
        self.assertIsInstance(fq.granularity, PerTensor)

    def test_init_with_per_axis_granularity(self):
        """Test initialization with PerAxis granularity."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerAxis(0),
        )
        self.assertIsInstance(fq.granularity, PerAxis)
        self.assertEqual(fq.granularity.axis, 0)

    def test_init_with_per_group_granularity(self):
        """Test initialization with PerGroup granularity."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(128),
        )
        self.assertIsInstance(fq.granularity, PerGroup)
        self.assertEqual(fq.granularity.group_size, 128)

    def test_init_with_dynamic_quantization(self):
        """Test initialization with is_dynamic=True."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
            is_dynamic=True,
        )
        self.assertTrue(fq.is_dynamic)


class AffineFakeQuantizeForwardTest(unittest.TestCase):
    """Test forward pass behavior of AffineFakeQuantize."""

    def test_forward_observer_enabled(self):
        """Test that observer updates statistics when enabled."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        x = torch.randn(10, 20)
        fq(x)

        self.assertIsNotNone(fq.block_size)
        self.assertTrue(hasattr(fq.activation_post_process, "min_val"))
        self.assertTrue(hasattr(fq.activation_post_process, "max_val"))

    def test_forward_observer_disabled(self):
        """Test that observer does not update when disabled."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        x = torch.randn(10, 20)
        fq(x)  # First forward to set scale/zp/block_size

        initial_scale = fq.scale.clone()
        fq.disable_observer()

        x2 = torch.randn(10, 20) * 100
        fq(x2)

        torch.testing.assert_close(fq.scale, initial_scale)

    def test_forward_fake_quant_enabled(self):
        """Test that fake quant is applied when enabled."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        x = torch.randn(10, 20)
        y = fq(x)

        self.assertFalse(torch.equal(x, y))
        self.assertEqual(x.shape, y.shape)

    def test_forward_fake_quant_disabled(self):
        """Test that input passes through unchanged when fake quant is disabled."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        x = torch.randn(10, 20)
        fq(x)
        fq.disable_fake_quant()

        x2 = torch.randn(10, 20)
        y2 = fq(x2)

        torch.testing.assert_close(x2, y2)

    def test_forward_output_shape_matches_input(self):
        """Test that output shape matches input shape for all granularities."""
        granularities = [
            PerTensor(),
            PerAxis(0),
            PerGroup(64),
        ]

        for granularity in granularities:
            with self.subTest(granularity=type(granularity).__name__):
                fq = AffineFakeQuantize(
                    observer=AffineQuantizedMinMaxObserver,
                    mapping_type=MappingType.SYMMETRIC,
                    target_dtype=torch.int8,
                    granularity=granularity,
                )

                x = torch.randn(10, 256)
                y = fq(x)

                self.assertEqual(x.shape, y.shape)


class AffineFakeQuantizeGroupwiseTest(unittest.TestCase):
    """Test groupwise quantization with AffineFakeQuantize."""

    def test_groupwise_scale_shape(self):
        """Test that scale shape is correct for groupwise quantization."""
        group_size = 64
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(group_size),
        )

        out_features, in_features = 32, 256
        x = torch.randn(out_features, in_features)
        fq(x)

        expected_scale_shape = (out_features, in_features // group_size)
        self.assertEqual(fq.scale.shape, expected_scale_shape)

    def test_groupwise_zero_point_shape(self):
        """Test that zero_point shape is correct for groupwise quantization."""
        group_size = 128
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(group_size),
        )

        out_features, in_features = 64, 512
        x = torch.randn(out_features, in_features)
        fq(x)

        expected_zp_shape = (out_features, in_features // group_size)
        self.assertEqual(fq.zero_point.shape, expected_zp_shape)

    def test_groupwise_output_matches_direct_fake_quant(self):
        """Test that AffineFakeQuantize output matches direct _fake_quantize_affine call."""
        group_size = 64
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(group_size),
        )

        x = torch.randn(32, 256)
        y_fq = fq(x)

        y_direct = _fake_quantize_affine(
            x,
            fq.block_size,
            fq.scale,
            fq.zero_point,
            fq.target_dtype,
            fq.quant_min,
            fq.quant_max,
            fq.zero_point_domain,
        )

        torch.testing.assert_close(y_fq, y_direct)


class AffineFakeQuantizeStateDictTest(unittest.TestCase):
    """Test state dict save/load functionality."""

    def test_save_load_state_dict(self):
        """Test that scale and zero_point are preserved through save/load."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        x = torch.randn(10, 20)
        fq(x)

        scale_before = fq.scale.clone()
        zp_before = fq.zero_point.clone()

        state_dict = fq.state_dict()

        fq2 = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )
        fq2.load_state_dict(state_dict)

        torch.testing.assert_close(fq2.scale, scale_before)
        torch.testing.assert_close(fq2.zero_point, zp_before)

    def test_save_load_groupwise_state_dict(self):
        """Test state dict save/load with groupwise quantization."""
        group_size = 64
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(group_size),
        )

        x = torch.randn(32, 256)
        fq(x)

        scale_before = fq.scale.clone()
        zp_before = fq.zero_point.clone()

        state_dict = fq.state_dict()

        fq2 = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(group_size),
        )
        fq2.load_state_dict(state_dict)

        torch.testing.assert_close(fq2.scale, scale_before)
        torch.testing.assert_close(fq2.zero_point, zp_before)


class AffineFakeQuantizeGradientTest(unittest.TestCase):
    """Test gradient flow through AffineFakeQuantize (STE behavior)."""

    def test_gradient_flow(self):
        """Test that gradients flow through fake quantize (Straight-Through Estimator)."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        x = torch.randn(10, 20, requires_grad=True)
        y = fq(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_qat_training_loop(self):
        """Test AffineFakeQuantize in a simple QAT training loop."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(256, 128)
                self.weight_fq = AffineFakeQuantize(
                    observer=AffineQuantizedMinMaxObserver,
                    mapping_type=MappingType.SYMMETRIC,
                    target_dtype=torch.int8,
                    granularity=PerGroup(64),
                )
                self.act_fq = AffineFakeQuantize(
                    observer=AffineQuantizedMinMaxObserver,
                    mapping_type=MappingType.ASYMMETRIC,
                    target_dtype=torch.uint8,
                    granularity=PerTensor(),
                )

            def forward(self, x):
                x = self.act_fq(x)
                weight = self.weight_fq(self.linear.weight)
                x = torch.nn.functional.linear(x, weight, self.linear.bias)
                return x

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(3):
            x = torch.randn(8, 256)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class AffineFakeQuantizeEnableDisableTest(unittest.TestCase):
    """Test enable/disable functionality for observer and fake_quant."""

    def test_enable_disable_fake_quant(self):
        """Test enable_fake_quant and disable_fake_quant methods."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        self.assertEqual(fq.fake_quant_enabled[0], 1)

        fq.disable_fake_quant()
        self.assertEqual(fq.fake_quant_enabled[0], 0)

        fq.enable_fake_quant()
        self.assertEqual(fq.fake_quant_enabled[0], 1)

    def test_enable_disable_observer(self):
        """Test enable_observer and disable_observer methods."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=torch.uint8,
            granularity=PerTensor(),
        )

        self.assertEqual(fq.observer_enabled[0], 1)

        fq.disable_observer()
        self.assertEqual(fq.observer_enabled[0], 0)

        fq.enable_observer()
        self.assertEqual(fq.observer_enabled[0], 1)


class DefaultAffineFakeQuantTest(unittest.TestCase):
    """Test default factory functions."""

    def test_default_affine_fake_quant(self):
        """Test default_affine_fake_quant configuration."""
        fq = default_affine_fake_quant()
        self.assertEqual(fq.target_dtype, torch.uint8)
        self.assertEqual(fq.mapping_type, MappingType.ASYMMETRIC)
        self.assertIsInstance(fq.granularity, PerTensor)

    def test_default_groupwise_fake_quant(self):
        """Test default_groupwise_fake_quant configuration."""
        fq = default_groupwise_fake_quant()
        self.assertEqual(fq.target_dtype, torch.int8)
        self.assertEqual(fq.mapping_type, MappingType.SYMMETRIC)
        self.assertIsInstance(fq.granularity, PerGroup)
        self.assertEqual(fq.granularity.group_size, 128)

    def test_default_affine_per_axis_fake_quant(self):
        """Test default_affine_per_axis_fake_quant configuration."""
        fq = default_affine_per_axis_fake_quant()
        self.assertEqual(fq.target_dtype, torch.int8)
        self.assertEqual(fq.mapping_type, MappingType.SYMMETRIC)
        self.assertIsInstance(fq.granularity, PerAxis)
        self.assertEqual(fq.granularity.axis, 0)


class AffineFakeQuantizeExtraReprTest(unittest.TestCase):
    """Test extra_repr method."""

    def test_extra_repr_contains_all_attributes(self):
        """Test that extra_repr contains all relevant attributes."""
        fq = AffineFakeQuantize(
            observer=AffineQuantizedMinMaxObserver,
            mapping_type=MappingType.SYMMETRIC,
            target_dtype=torch.int8,
            granularity=PerGroup(128),
        )

        x = torch.randn(10, 256)
        fq(x)

        repr_str = fq.extra_repr()

        self.assertIn("fake_quant_enabled", repr_str)
        self.assertIn("observer_enabled", repr_str)
        self.assertIn("target_dtype", repr_str)
        self.assertIn("granularity", repr_str)
        self.assertIn("mapping_type", repr_str)
        self.assertIn("scale", repr_str)
        self.assertIn("zero_point", repr_str)
        self.assertIn("block_size", repr_str)


# --- Shared Test Components for QAT End-to-End Tests ---


class EmbeddingLinearModel(torch.nn.Module):
    """A simple model with embedding + linear for QAT testing."""

    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len) -> indices for embedding
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.mean(dim=1)  # (batch_size, embedding_dim) - simple pooling
        x = self.linear(x)  # (batch_size, output_dim)
        return x


class EmbeddingQuantizer(Quantizer):
    """
    A configurable embedding quantizer for QAT testing.

    This quantizer annotates embedding operations with a configurable
    observer or fake quantize constructor.

    Args:
        observer_or_fake_quant_ctr: The observer or fake quantize constructor to use
            for annotating embedding weights.
    """

    def __init__(self, observer_or_fake_quant_ctr):
        super().__init__()
        self._observer_or_fake_quant_ctr = observer_or_fake_quant_ctr

    def annotate(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.target != torch.ops.aten.embedding.default:
                continue

            annotation = node.meta.get(
                "quantization_annotation",
                QuantizationAnnotation(_annotated=True),
            )
            annotation.input_qspec_map[node.args[0]] = QuantizationSpec(
                dtype=torch.int8,
                quant_min=-8,
                quant_max=7,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
                observer_or_fake_quant_ctr=self._observer_or_fake_quant_ctr,
            )
            node.meta["quantization_annotation"] = annotation

    def validate(self, graph_module):
        pass


class EmbeddingQuantizerWithObserver(EmbeddingQuantizer):
    """Embedding quantizer that uses AffineQuantizedMinMaxObserver directly."""

    def __init__(self):
        super().__init__(
            AffineQuantizedMinMaxObserver.with_args(
                mapping_type=MappingType.SYMMETRIC,
                target_dtype=torch.int8,
                granularity=PerGroup(32),
                block_size=(1, 32),
            )
        )


class EmbeddingQuantizerWithAffineFakeQuantize(EmbeddingQuantizer):
    """
    An embedding quantizer that uses AffineFakeQuantize with AffineQuantizedMinMaxObserver.
    Similar to CPUEmbeddingQuantizer but uses the new affine fake quantize infrastructure.
    """

    def __init__(self):
        super().__init__(
            AffineFakeQuantize.with_args(
                observer=AffineQuantizedMinMaxObserver,
                mapping_type=MappingType.SYMMETRIC,
                target_dtype=torch.int8,
                granularity=PerGroup(32),
            )
        )


class AffineFakeQuantizeQATEndToEndTest(unittest.TestCase):
    """Test QAT workflow with AffineFakeQuantize using embedding + linear model."""

    # Default test model parameters
    VOCAB_SIZE = 100
    EMBEDDING_DIM = 64
    OUTPUT_DIM = 32
    BATCH_SIZE = 2
    SEQ_LEN = 8

    def _run_qat_embedding_linear_workflow(
        self, embedding_quantizer, check_affine_fake_quantize=False
    ):
        """
        Common QAT workflow for embedding + linear model testing.

        Args:
            embedding_quantizer: The quantizer to use for embedding quantization.
            check_affine_fake_quantize: If True, verify that AffineFakeQuantize
                modules are inserted in the prepared model.

        Returns:
            Tuple of (prepared_model, converted_model) for additional assertions.
        """
        # Create the model
        model = EmbeddingLinearModel(
            self.VOCAB_SIZE, self.EMBEDDING_DIM, self.OUTPUT_DIM
        )

        # Create example inputs
        example_inputs = (
            torch.randint(0, self.VOCAB_SIZE, (self.BATCH_SIZE, self.SEQ_LEN)),
        )

        # Create composable quantizer with:
        # 1. Provided embedding quantizer
        # 2. XNNPACKQuantizer for linear layer
        xnnpack_quantizer = XNNPACKQuantizer()
        xnnpack_quantizer.set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
        )

        composable_quantizer = ComposableQuantizer(
            [embedding_quantizer, xnnpack_quantizer]
        )

        # Export the model
        exported_model = torch.export.export(
            model, example_inputs, strict=True
        ).module()

        # Prepare for QAT
        prepared_model = prepare_qat_pt2e(exported_model, composable_quantizer)

        # Run forward for 1 step to verify QAT model works
        output = prepared_model(*example_inputs)
        self.assertEqual(output.shape, (self.BATCH_SIZE, self.OUTPUT_DIM))

        # Optionally verify AffineFakeQuantize insertion
        if check_affine_fake_quantize:
            has_affine_fake_quantize = any(
                isinstance(module, AffineFakeQuantize)
                for module in prepared_model.modules()
            )
            self.assertTrue(
                has_affine_fake_quantize,
                "Expected AffineFakeQuantize to be inserted in the model for embedding quantization",
            )

        # Convert to quantized model
        converted_model = convert_pt2e(prepared_model)

        # Verify converted model can run forward
        converted_output = converted_model(*example_inputs)
        self.assertEqual(converted_output.shape, (self.BATCH_SIZE, self.OUTPUT_DIM))

        return prepared_model, converted_model

    def test_qat_embedding_linear_with_observer_and_composable_quantizer(self):
        """
        Test QAT workflow of a toy model with 1 embedding layer followed by 1 linear layer.
        Uses a ComposableQuantizer with:
        - EmbeddingQuantizerWithObserver for embedding (uses observer directly)
        - XNNPACKQuantizer for linear layer

        This test verifies the full workflow:
        1. Model can be exported with torch.export
        2. QAT preparation with prepare_qat_pt2e works
        3. Forward pass runs successfully after QAT preparation
        4. Conversion with convert_pt2e completes successfully
        """
        self._run_qat_embedding_linear_workflow(
            embedding_quantizer=EmbeddingQuantizerWithObserver(),
            check_affine_fake_quantize=False,
        )

    def test_qat_embedding_linear_with_composable_quantizer(self):
        """
        Test QAT workflow of a toy model with 1 embedding layer followed by 1 linear layer.
        Uses a ComposableQuantizer with:
        - EmbeddingQuantizerWithAffineFakeQuantize for embedding (uses AffineFakeQuantize)
        - XNNPACKQuantizer for linear layer

        This test verifies:
        1. Model can be exported with torch.export
        2. QAT preparation with prepare_qat_pt2e works
        3. Forward pass runs successfully after QAT preparation
        4. Conversion with convert_pt2e completes successfully
        """
        self._run_qat_embedding_linear_workflow(
            embedding_quantizer=EmbeddingQuantizerWithAffineFakeQuantize(),
            check_affine_fake_quantize=True,
        )


if __name__ == "__main__":
    unittest.main()
