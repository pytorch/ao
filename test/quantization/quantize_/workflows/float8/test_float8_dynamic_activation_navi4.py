# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Float8 dynamic activation quantization on AMD Navi4 (gfx1200/gfx1201) GPUs.

This test suite validates that Float8DynamicActivationFloat8WeightConfig works correctly
on Navi4 hardware (e.g. Radeon AI PRO R9700) running ROCm 7.2+, which supports float8
via torch._scaled_mm but was previously blocked by a hardware guard that only allowed
CUDA SM>=8.9 and MI300+ (gfx940/941/942).

Hardware requirements for Float8 dynamic activation:
  - NVIDIA: SM >= 8.9 (Ada Lovelace / Hopper and later)
  - AMD:    MI300+ (gfx940/941/942) or Navi4+ (gfx1200/gfx1201)
  - Intel:  XPU

Float8WeightOnlyConfig runs on any accelerator (no hardware guard).
"""

import copy
import io
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Float8Tensor,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import (
    get_current_accelerator_device,
    is_MI300,
    is_Navi4,
    is_sm_at_least_89,
)

# ---------------------------------------------------------------------------
# Helper: True when the current GPU supports Float8 dynamic activation quant
# ---------------------------------------------------------------------------
_SUPPORTS_FLOAT8_DYNAMIC = is_sm_at_least_89() or is_MI300() or is_Navi4()

# Minimum SQNR (dB) we require from all quantized models
_MIN_SQNR_DB = 20.0


# ---------------------------------------------------------------------------
# Shared model definitions
# ---------------------------------------------------------------------------


class TwoLinearModel(nn.Module):
    """Two stacked linear layers; a minimal proxy for a feed-forward block."""

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 512,
        bias: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.fc2 = nn.Linear(out_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class TransformerBlock(nn.Module):
    """Minimal transformer block with pre-norm and FFN."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_proj(self.norm1(x))
        x = x + self.ff2(torch.relu(self.ff1(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Test suite A: Float8WeightOnly — runs on any accelerator
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not torch.accelerator.is_available(), "skipping when no accelerator is available"
)
class TestFloat8WeightOnly(TorchAOIntegrationTestCase):
    """
    Tests for Float8WeightOnlyConfig.

    Weight-only quantization pre-quantizes weights to float8 offline; activations
    remain in the original dtype (bfloat16 or float32) at runtime.  This path has
    no hardware guard and runs on any CUDA/ROCm/XPU accelerator.
    """

    def setUp(self):
        self.device = get_current_accelerator_device()
        torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    # scale shape
    # ------------------------------------------------------------------

    def test_weight_scale_shape(self):
        """Weight scales produced by Float8WeightOnlyConfig must be 2-D tensors
        with a positive number of rows (PerRow default) and singleton column."""
        N, K = 512, 256
        model = TwoLinearModel(in_features=K, out_features=N)
        model = model.to(torch.bfloat16).to(self.device).eval()
        quantize_(model, Float8WeightOnlyConfig())

        self.assertIsInstance(model.fc1.weight, Float8Tensor)
        self.assertIsInstance(model.fc2.weight, Float8Tensor)
        # PerRow default: (out_features, 1)
        self.assertGreaterEqual(model.fc1.weight.scale.shape[0], 1)
        self.assertEqual(model.fc1.weight.scale.ndim, 2)

    # ------------------------------------------------------------------
    # quantization accuracy
    # ------------------------------------------------------------------

    def _assert_sqnr(self, original: nn.Module, quantized: nn.Module, x: torch.Tensor):
        with torch.inference_mode():
            sqnr = compute_error(original(x), quantized(x))
        self.assertGreater(
            sqnr,
            _MIN_SQNR_DB,
            f"SQNR {sqnr:.2f} dB is below the required {_MIN_SQNR_DB} dB threshold",
        )

    def test_sqnr_bfloat16(self):
        """Float8WeightOnly must achieve SQNR > 20 dB for bfloat16 inputs."""
        model = TwoLinearModel(1024, 1024).to(torch.bfloat16).to(self.device).eval()
        q_model = copy.deepcopy(model)
        quantize_(q_model, Float8WeightOnlyConfig())
        x = torch.randn(8, 1024, dtype=torch.bfloat16, device=self.device)
        self._assert_sqnr(model, q_model, x)

    def test_sqnr_float32(self):
        """Float8WeightOnly must achieve SQNR > 20 dB for float32 inputs."""
        model = TwoLinearModel(1024, 1024).to(torch.float32).to(self.device).eval()
        q_model = copy.deepcopy(model)
        quantize_(q_model, Float8WeightOnlyConfig())
        x = torch.randn(8, 1024, dtype=torch.float32, device=self.device)
        self._assert_sqnr(model, q_model, x)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def test_state_dict_round_trip(self):
        """A Float8WeightOnly model must survive a state_dict save/load cycle
        and produce identical outputs."""
        model = TwoLinearModel().to(torch.bfloat16).to(self.device).eval()
        q_model = copy.deepcopy(model)
        quantize_(q_model, Float8WeightOnlyConfig())

        buf = io.BytesIO()
        torch.save(q_model.state_dict(), buf)
        buf.seek(0)

        restored = copy.deepcopy(model)
        quantize_(restored, Float8WeightOnlyConfig())
        restored.load_state_dict(torch.load(buf, weights_only=True))

        x = torch.randn(4, 512, dtype=torch.bfloat16, device=self.device)
        with torch.inference_mode():
            self.assertTrue(
                torch.allclose(q_model(x), restored(x), atol=1e-3),
                "Outputs differ after state_dict round-trip",
            )

    # ------------------------------------------------------------------
    # torch.compile
    # ------------------------------------------------------------------

    def test_compile_eager_consistency(self):
        """torch.compile(fullgraph=True) must produce outputs consistent with
        eager mode (atol=1e-3 for weight-only path)."""
        model = TwoLinearModel().to(torch.bfloat16).to(self.device).eval()
        quantize_(model, Float8WeightOnlyConfig())
        compiled = torch.compile(model, fullgraph=True)

        x = torch.randn(4, 512, dtype=torch.bfloat16, device=self.device)
        with torch.inference_mode():
            self.assertTrue(
                torch.allclose(model(x), compiled(x), atol=1e-3),
                "Compiled and eager outputs differ",
            )

    # ------------------------------------------------------------------
    # batch size robustness
    # ------------------------------------------------------------------

    def test_multiple_batch_sizes(self):
        """Quantized model must produce correctly shaped outputs for a range of
        batch sizes without errors."""
        model = TwoLinearModel().to(torch.bfloat16).to(self.device).eval()
        quantize_(model, Float8WeightOnlyConfig())
        for batch in (1, 4, 16, 64):
            x = torch.randn(batch, 512, dtype=torch.bfloat16, device=self.device)
            with torch.inference_mode():
                out = model(x)
            self.assertEqual(out.shape, (batch, 512))

    # ------------------------------------------------------------------
    # selective quantization
    # ------------------------------------------------------------------

    def test_filter_fn_selective_quantization(self):
        """filter_fn must restrict quantization to matching layers only."""

        class TwoLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.keep_bf16 = nn.Linear(512, 512, bias=False)
                self.quantize_this = nn.Linear(512, 512, bias=False)

            def forward(self, x):
                return self.quantize_this(self.keep_bf16(x))

        model = TwoLayer().to(torch.bfloat16).to(self.device).eval()
        quantize_(
            model,
            Float8WeightOnlyConfig(),
            filter_fn=lambda mod, fqn: "quantize_this" in fqn,
        )
        self.assertNotIsInstance(
            model.keep_bf16.weight,
            Float8Tensor,
            "keep_bf16 layer must NOT be quantized",
        )
        self.assertIsInstance(
            model.quantize_this.weight,
            Float8Tensor,
            "quantize_this layer must be quantized",
        )

    # ------------------------------------------------------------------
    # transformer block
    # ------------------------------------------------------------------

    def test_transformer_block_selective_layers(self):
        """Only nn.Linear layers should be quantized; LayerNorm weights must
        remain in their original dtype."""
        block = TransformerBlock().to(torch.bfloat16).to(self.device).eval()
        original = copy.deepcopy(block)
        quantize_(block, Float8WeightOnlyConfig())

        self.assertNotIsInstance(
            block.norm1.weight,
            Float8Tensor,
            "LayerNorm weight must not be quantized",
        )
        self.assertIsInstance(block.ff1.weight, Float8Tensor)
        self.assertIsInstance(block.ff2.weight, Float8Tensor)

        x = torch.randn(4, 32, 512, dtype=torch.bfloat16, device=self.device)
        self._assert_sqnr(original, block, x)

    # ------------------------------------------------------------------
    # Float8Tensor properties
    # ------------------------------------------------------------------

    def test_float8tensor_properties(self):
        """Float8Tensor must expose the correct dtype, scale, and logical shape."""
        model = TwoLinearModel(256, 128).to(torch.bfloat16).to(self.device).eval()
        quantize_(model, Float8WeightOnlyConfig())

        w = model.fc1.weight
        self.assertIsInstance(w, Float8Tensor)
        self.assertEqual(
            w.qdata.dtype,
            torch.float8_e4m3fn,
            "Quantized data must be stored as float8_e4m3fn",
        )
        self.assertTrue(w.scale.is_floating_point(), "Scale must be a floating-point tensor")
        self.assertTrue((w.scale > 0).all(), "All scale values must be positive")
        self.assertEqual(
            w.shape,
            (128, 256),
            "Logical shape of quantized weight must match original weight shape",
        )


# ---------------------------------------------------------------------------
# Test suite B: Float8DynamicActivation — requires SM89+ / MI300+ / Navi4+
# ---------------------------------------------------------------------------

_SKIP_NO_FLOAT8_DYNAMIC = unittest.skipIf(
    not _SUPPORTS_FLOAT8_DYNAMIC,
    "Float8 dynamic activation quantization requires SM>=8.9, MI300+, or Navi4+",
)


@unittest.skipIf(
    not torch.accelerator.is_available(), "skipping when no accelerator is available"
)
@_SKIP_NO_FLOAT8_DYNAMIC
class TestFloat8DynamicActivation(TorchAOIntegrationTestCase):
    """
    Tests for Float8DynamicActivationFloat8WeightConfig.

    Both weights and activations are quantized to float8 at runtime.
    Requires hardware that supports float8 matrix multiplication:
      - NVIDIA SM >= 8.9 (Ada / Hopper)
      - AMD MI300+ (gfx940/941/942)
      - AMD Navi4+ (gfx1200/gfx1201)
    """

    def setUp(self):
        self.device = get_current_accelerator_device()
        torch.set_grad_enabled(False)

    def _make_model(
        self,
        in_features: int = 512,
        out_features: int = 512,
        dtype: torch.dtype = torch.bfloat16,
    ) -> TwoLinearModel:
        return (
            TwoLinearModel(in_features, out_features)
            .to(dtype)
            .to(self.device)
            .eval()
        )

    def _assert_sqnr(self, original: nn.Module, quantized: nn.Module, x: torch.Tensor):
        with torch.inference_mode():
            sqnr = compute_error(original(x), quantized(x))
        self.assertGreater(
            sqnr,
            _MIN_SQNR_DB,
            f"SQNR {sqnr:.2f} dB is below the required {_MIN_SQNR_DB} dB threshold",
        )

    # ------------------------------------------------------------------
    # PerTensor granularity
    # ------------------------------------------------------------------

    def test_per_tensor_scale_shape(self):
        """PerTensor quantization must produce a (1, 1) scale tensor per weight."""
        N, K = 512, 256
        model = self._make_model(K, N)
        q_model = copy.deepcopy(model)
        quantize_(
            q_model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        )
        self.assertIsInstance(q_model.fc1.weight, Float8Tensor)
        self.assertEqual(q_model.fc1.weight.scale.shape, (1, 1))
        self.assertEqual(q_model.fc2.weight.scale.shape, (1, 1))

    def test_per_tensor_sqnr(self):
        """PerTensor dynamic activation quantization must achieve SQNR > 20 dB."""
        model = self._make_model(1024, 1024)
        q_model = copy.deepcopy(model)
        quantize_(
            q_model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        )
        x = torch.randn(8, 1024, dtype=torch.bfloat16, device=self.device)
        self._assert_sqnr(model, q_model, x)

    # ------------------------------------------------------------------
    # PerRow granularity
    # ------------------------------------------------------------------

    def test_per_row_scale_shape(self):
        """PerRow quantization must produce (out_features, 1) scale tensors."""
        N, K = 512, 512
        model = self._make_model(K, N)
        q_model = copy.deepcopy(model)
        quantize_(
            q_model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        )
        # fc1 weight shape: (N, K) -> scale shape: (N, 1)
        self.assertEqual(q_model.fc1.weight.scale.shape, (N, 1))
        # fc2 weight shape: (K, N) -> scale shape: (K, 1)
        self.assertEqual(q_model.fc2.weight.scale.shape, (K, 1))

    def test_per_row_sqnr(self):
        """PerRow dynamic activation quantization must achieve SQNR > 20 dB."""
        N = K = 512
        model = self._make_model(K, N)
        q_model = copy.deepcopy(model)
        quantize_(
            q_model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        )
        x = torch.randn(4, K, dtype=torch.bfloat16, device=self.device)
        self._assert_sqnr(model, q_model, x)

    # ------------------------------------------------------------------
    # torch.compile
    # ------------------------------------------------------------------

    def test_compile_eager_consistency(self):
        """torch.compile(fullgraph=True) output must be close to eager output.

        Float8 dynamic activation kernels may reorder floating-point operations
        during compilation, so a slightly relaxed atol=0.05 is used instead of
        the tighter 1e-3 used for weight-only quantization.
        """
        model = self._make_model()
        q_model = copy.deepcopy(model)
        quantize_(
            q_model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        )
        compiled = torch.compile(q_model, fullgraph=True)

        x = torch.randn(4, 512, dtype=torch.bfloat16, device=self.device)
        with torch.inference_mode():
            out_eager = q_model(x)
            out_compiled = compiled(x)
        max_diff = (out_eager - out_compiled).abs().max().item()
        self.assertLess(
            max_diff,
            0.05,
            f"Compiled and eager outputs differ by {max_diff:.4f} (atol=0.05)",
        )

    # ------------------------------------------------------------------
    # activation value clamping
    # ------------------------------------------------------------------

    def test_activation_value_bounds(self):
        """activation_value_lb/ub must not introduce NaN or Inf in the output."""
        model = self._make_model()
        q_model = copy.deepcopy(model)
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            activation_value_lb=-10.0,
            activation_value_ub=10.0,
        )
        quantize_(q_model, config)
        x = torch.randn(4, 512, dtype=torch.bfloat16, device=self.device)
        with torch.inference_mode():
            out = q_model(x)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")

    # ------------------------------------------------------------------
    # input shape robustness
    # ------------------------------------------------------------------

    def test_multiple_batch_sizes(self):
        """Quantized model must handle batch sizes 1 / 4 / 16 / 64."""
        model = self._make_model()
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        )
        for batch in (1, 4, 16, 64):
            x = torch.randn(batch, 512, dtype=torch.bfloat16, device=self.device)
            with torch.inference_mode():
                out = model(x)
            self.assertEqual(out.shape, (batch, 512))

    def test_3d_input(self):
        """Quantized linear layers must accept 3-D inputs (seq_len, batch, feat)
        as used in typical Transformer inference."""
        model = self._make_model()
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        )
        x = torch.randn(32, 4, 512, dtype=torch.bfloat16, device=self.device)
        with torch.inference_mode():
            out = model(x)
        self.assertEqual(out.shape, (32, 4, 512))


# ---------------------------------------------------------------------------
# Test suite C: raw float8 matmul — validates Navi4 hardware capability
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not torch.accelerator.is_available(), "skipping when no accelerator is available"
)
@_SKIP_NO_FLOAT8_DYNAMIC
class TestFloat8RawMatmul(TorchAOIntegrationTestCase):
    """
    Validates that torch._scaled_mm (the underlying float8 kernel) is
    functional and numerically accurate on the current GPU.

    This serves as a sanity check that the hardware actually supports float8
    before running the higher-level quantization tests.
    """

    def setUp(self):
        self.device = get_current_accelerator_device()
        torch.set_grad_enabled(False)

    def test_scaled_mm_accuracy(self):
        """torch._scaled_mm with PerTensor scaling must achieve SQNR > 20 dB
        compared to a bfloat16 reference matmul."""
        M, N, K = 128, 256, 512
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        a_bf16 = torch.randn(M, K, device=self.device, dtype=torch.bfloat16)
        b_bf16 = torch.randn(N, K, device=self.device, dtype=torch.bfloat16)

        a_scale = (a_bf16.abs().max() / fp8_max).to(torch.float32).reshape(1, 1)
        b_scale = (b_bf16.abs().max() / fp8_max).to(torch.float32).reshape(1, 1)

        a_f8 = (a_bf16 * (fp8_max / a_bf16.abs().max())).to(torch.float8_e4m3fn)
        b_f8 = (b_bf16 * (fp8_max / b_bf16.abs().max())).to(torch.float8_e4m3fn)

        out_f8 = torch._scaled_mm(
            a_f8, b_f8.t(),
            scale_a=a_scale,
            scale_b=b_scale,
            out_dtype=torch.bfloat16,
        )
        out_ref = torch.mm(a_bf16, b_bf16.t())

        self.assertEqual(out_f8.shape, (M, N))
        sqnr = compute_error(out_ref.float(), out_f8.float())
        self.assertGreater(
            sqnr,
            _MIN_SQNR_DB,
            f"_scaled_mm SQNR {sqnr:.2f} dB is below {_MIN_SQNR_DB} dB",
        )


# ---------------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()
