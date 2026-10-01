# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import unittest

import torch

from torchao.prototype.qat import (
    CodebookFakeQuantizeConfig,
    CodebookFakeQuantizer,
)
from torchao.prototype.quantization.codebook_coreml import (
    CodebookQuantizedTensor,
    CodebookWeightOnlyConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig
from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs
from torchao.quantization.qat.linear import FakeQuantizedLinear
from torchao.utils import is_package_at_least


class TestCodebookQATConfig(unittest.TestCase):
    """
    Tests for the codebook QAT config: PTQ->QAT config inference, the prepare
    module swap, config validation, and checkpoint save/load of the cached
    codebook. These do not run k-means and so do not require coremltools.
    """

    def test_infer_codebook_config(self):
        base_config = CodebookWeightOnlyConfig(dtype=torch.uint4, block_size=[-1, 16])
        (act_config, weight_config) = _infer_fake_quantize_configs(base_config)
        self.assertIsNone(act_config)
        self.assertIsInstance(weight_config, CodebookFakeQuantizeConfig)
        self.assertEqual(weight_config.dtype, torch.uint4)
        self.assertEqual(weight_config.block_size, [-1, 16])

    def test_prepare_swaps_to_fake_quantized_linear(self):
        base_config = CodebookWeightOnlyConfig(dtype=torch.uint4, block_size=[-1, 16])
        m = torch.nn.Sequential(torch.nn.Linear(64, 64))
        quantize_(m, QATConfig(base_config, step="prepare"))
        self.assertIsInstance(m[0], FakeQuantizedLinear)
        self.assertIsNone(m[0].activation_fake_quantizer)
        self.assertIsInstance(m[0].weight_fake_quantizer, CodebookFakeQuantizer)

    def test_config_validation(self):
        # unsupported dtype
        with self.assertRaises(ValueError):
            CodebookFakeQuantizeConfig(dtype=torch.int8)
        # invalid refresh interval
        with self.assertRaises(ValueError):
            CodebookFakeQuantizeConfig(dtype=torch.uint4, refresh_interval=0)

    def test_config_default_refresh_interval(self):
        config = CodebookFakeQuantizeConfig(dtype=torch.uint4)
        self.assertEqual(config.refresh_interval, 100)

    def test_codebook_saved_and_loaded_from_checkpoint(self):
        """
        The cached codebook is a persistent buffer, so it is saved in the
        state_dict and restored on load (into a fresh module whose codebook is
        still the lazily-created None). No coremltools needed: we set the codebook
        directly to exercise the save/load plumbing.
        """
        config = CodebookFakeQuantizeConfig(dtype=torch.uint4, block_size=[-1, 16])

        fake_quantizer = CodebookFakeQuantizer(config)
        # simulate a populated codebook: (g0, g1, 2**nbits, 1) = (1, 4, 16, 1)
        codebook = torch.randn(1, 4, 16, 1)
        fake_quantizer._codebook = codebook
        state_dict = fake_quantizer.state_dict()
        self.assertIn("_codebook", state_dict)

        # load into a fresh module whose codebook is still None
        loaded = CodebookFakeQuantizer(config)
        self.assertIsNone(loaded._codebook)
        loaded.load_state_dict(state_dict)
        torch.testing.assert_close(loaded._codebook, codebook)

    def test_deferred_convert_reuses_persisted_codebook(self):
        """
        Deferred lifecycle: prepare -> train -> save prepared checkpoint ->
        load later -> convert -> inference. Convert must reuse the persisted
        codebook (assignment only, no k-means), so this whole path needs no
        coremltools. We set the codebook directly to stand in for the codebook a
        training forward would have produced.
        """
        base_config = CodebookWeightOnlyConfig(dtype=torch.uint4, block_size=[-1, 16])

        def build():
            return torch.nn.Sequential(torch.nn.Linear(64, 16, bias=True))

        # prepare, then persist a codebook (as a training forward would have)
        m = build()
        quantize_(m, QATConfig(base_config, step="prepare"))
        persisted = torch.randn(1, 4, 16, 1)  # (g0=1, g1=4, 2**nbits=16, vec_dim=1)
        m[0].weight_fake_quantizer._codebook = persisted
        state_dict = copy.deepcopy(m.state_dict())

        # later: load the prepared checkpoint into a fresh prepared model
        m2 = build()
        quantize_(m2, QATConfig(base_config, step="prepare"))
        self.assertIsNone(m2[0].weight_fake_quantizer._codebook)
        m2.load_state_dict(state_dict)
        torch.testing.assert_close(m2[0].weight_fake_quantizer._codebook, persisted)

        # convert reuses the persisted codebook (no k-means / coremltools) ...
        quantize_(m2, QATConfig(base_config, step="convert"))
        converted = m2[0].weight
        self.assertIsInstance(converted, CodebookQuantizedTensor)
        torch.testing.assert_close(converted.codebook, persisted)

        # ... and inference runs
        out = m2(torch.randn(2, 64, dtype=torch.float32))
        self.assertEqual(out.shape, (2, 16))


@unittest.skipIf(
    not is_package_at_least("coremltools", "8.3.0"), "Requires coremltools >= 8.3.0"
)
class TestCodebookQAT(unittest.TestCase):
    # Small shapes + mostly uint4 (k=16 centroids) keep real k-means fast: the
    # weight below has few lookup tables per config, so each test runs in well
    # under a second even with coremltools installed.
    def setUp(self):
        torch.manual_seed(123)
        # (16, 64): rows divisible by row block (4), cols divisible by col block (16)
        self.weight = torch.randn(16, 64, dtype=torch.float32)
        self.in_features = 64
        self.out_features = 16

    def _fake_quant_configs(self):
        return [
            (torch.uint4, [-1, 16]),  # column grouping, k=16
            (torch.uint4, [4, -1]),  # row grouping, k=16
        ]

    def test_fake_quantizer_matches_ptq(self):
        """
        The fake quantizer output must match the PTQ dequantized weight exactly.
        """
        for code_dtype, block_size in self._fake_quant_configs():
            config = CodebookFakeQuantizeConfig(dtype=code_dtype, block_size=block_size)
            fake_quantizer = CodebookFakeQuantizer(config)
            fq_weight = fake_quantizer(self.weight)

            ptq_tensor = CodebookQuantizedTensor.from_float(
                self.weight, code_dtype, block_size
            )
            ptq_weight = ptq_tensor.dequantize()

            self.assertEqual(fq_weight.shape, self.weight.shape)
            # Matches PTQ (deterministic k-means, same underlying ops)
            torch.testing.assert_close(
                fq_weight.detach(),
                ptq_weight,
                msg=f"fake quant != PTQ for dtype={code_dtype}, block_size={block_size}",
            )

    def test_periodic_refresh(self):
        """
        The codebook should be recomputed via k-means only every
        `refresh_interval` steps (including the first step).
        """
        config = CodebookFakeQuantizeConfig(
            dtype=torch.uint4, block_size=[-1, 16], refresh_interval=3
        )
        fake_quantizer = CodebookFakeQuantizer(config)
        for _ in range(7):
            fake_quantizer(self.weight)
        # refreshes happen at steps 0, 3, 6
        self.assertEqual(fake_quantizer._step, 7)
        self.assertEqual(fake_quantizer._num_refreshes, 3)

    def test_refresh_interval_1_matches_ptq_every_step(self):
        """
        With refresh_interval=1 the codebook is recomputed every step, so every
        forward matches PTQ exactly.
        """
        config = CodebookFakeQuantizeConfig(
            dtype=torch.uint4, block_size=[-1, 16], refresh_interval=1
        )
        fake_quantizer = CodebookFakeQuantizer(config)
        ptq_weight = CodebookQuantizedTensor.from_float(
            self.weight, torch.uint4, [-1, 16]
        ).dequantize()
        for _ in range(3):
            fq_weight = fake_quantizer(self.weight)
            torch.testing.assert_close(fq_weight.detach(), ptq_weight)
        self.assertEqual(fake_quantizer._num_refreshes, 3)

    def test_loaded_codebook_used_without_reclustering(self):
        """
        A codebook restored from a checkpoint must be used as-is on the next
        forward, without re-running k-means (which would defeat the point of
        saving it and would require coremltools at inference time).
        """
        config = CodebookFakeQuantizeConfig(
            dtype=torch.uint4, block_size=[-1, 16], refresh_interval=1000
        )
        trained = CodebookFakeQuantizer(config)
        trained(self.weight)  # populate the codebook via k-means
        self.assertEqual(trained._num_refreshes, 1)

        loaded = CodebookFakeQuantizer(config)
        loaded.load_state_dict(trained.state_dict())

        out = loaded(self.weight)
        # used the loaded codebook (assignment path), did not re-run k-means
        self.assertEqual(loaded._num_refreshes, 0)
        torch.testing.assert_close(loaded._codebook, trained._codebook)
        # and the output matches the trained module's fake quant on the same weight
        torch.testing.assert_close(out.detach(), trained(self.weight).detach())

    def test_fake_quantizer_straight_through_gradient(self):
        """
        Gradients should flow to the original weight unchanged (straight-through).
        """
        config = CodebookFakeQuantizeConfig(dtype=torch.uint4, block_size=[-1, 16])
        fake_quantizer = CodebookFakeQuantizer(config)
        w = self.weight.clone().requires_grad_(True)
        out = fake_quantizer(w)
        out.sum().backward()
        self.assertIsNotNone(w.grad)
        # Straight-through estimator => gradient of sum() w.r.t. each element is 1
        torch.testing.assert_close(w.grad, torch.ones_like(w))

    def test_qat_prepare_and_convert_match_ptq(self):
        """
        End-to-end: without training, both the prepared (fake quantized) model
        and the converted model must be identical to running PTQ directly, since
        fake quant == PTQ dequant and convert (with no training) falls back to
        the PTQ handler.
        """
        for code_dtype, block_size in self._fake_quant_configs():
            base_config = CodebookWeightOnlyConfig(
                dtype=code_dtype, block_size=block_size
            )
            m = torch.nn.Sequential(
                torch.nn.Linear(self.in_features, self.out_features, bias=True)
            )
            m[0].weight = torch.nn.Parameter(self.weight.clone())
            example_inputs = (torch.randn(8, self.in_features, dtype=torch.float32),)
            msg = f"dtype={code_dtype}, block_size={block_size}"

            # PTQ baseline
            m_ptq = copy.deepcopy(m)
            quantize_(m_ptq, base_config)
            out_ptq = m_ptq(*example_inputs)

            # Prepare: prepared output matches PTQ output
            m_prepared = copy.deepcopy(m)
            quantize_(m_prepared, QATConfig(base_config, step="prepare"))
            torch.testing.assert_close(
                m_prepared(*example_inputs), out_ptq, msg=f"prepare != PTQ for {msg}"
            )

            # Convert (no training forward -> falls back to PTQ): converted weight
            # and output match PTQ
            m_converted = copy.deepcopy(m)
            quantize_(m_converted, QATConfig(base_config, step="prepare"))
            quantize_(m_converted, QATConfig(base_config, step="convert"))
            self.assertIsInstance(m_converted[0].weight, CodebookQuantizedTensor)
            torch.testing.assert_close(
                m_converted[0].weight.dequantize(),
                m_ptq[0].weight.dequantize(),
                msg=f"convert weight != PTQ for {msg}",
            )
            torch.testing.assert_close(
                m_converted(*example_inputs), out_ptq, msg=f"convert != PTQ for {msg}"
            )

    def test_convert_reuses_training_codebook(self):
        """
        After a training forward populates the cached codebook, convert must
        reuse that (frozen) codebook rather than re-clustering on the drifted
        final weights, so the deployed grid matches what training optimized
        against.
        """
        base_config = CodebookWeightOnlyConfig(dtype=torch.uint4, block_size=[-1, 16])
        m = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features, bias=True)
        )
        m[0].weight = torch.nn.Parameter(self.weight.clone())

        quantize_(m, QATConfig(base_config, step="prepare"))

        # Training forward populates the cached codebook (grid A from initial weights)
        m(torch.randn(8, self.in_features, dtype=torch.float32))
        cached_codebook = m[0].weight_fake_quantizer._codebook.clone()

        # Weights drift during "training"
        with torch.no_grad():
            m[0].weight.add_(0.5 * torch.randn_like(m[0].weight))
        final_weight = m[0].weight.detach().clone()

        quantize_(m, QATConfig(base_config, step="convert"))
        converted = m[0].weight
        self.assertIsInstance(converted, CodebookQuantizedTensor)

        # convert reused the training-time codebook, not a fresh re-cluster
        torch.testing.assert_close(converted.codebook, cached_codebook)

        # and it differs from re-clustering PTQ on the drifted final weights
        fresh = CodebookQuantizedTensor.from_float(final_weight, torch.uint4, [-1, 16])
        self.assertFalse(torch.equal(fresh.codebook, cached_codebook))


if __name__ == "__main__":
    unittest.main()
