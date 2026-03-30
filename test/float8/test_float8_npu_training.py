# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FP8 training on NPU via npu_quant_matmul.

Verifies that:
1. The NPU code path in addmm_float8_unwrapped is correctly dispatched
2. Float8Linear forward/backward produces valid results on NPU
3. convert_to_float8_training works end-to-end on NPU
4. The NPU path is actually invoked (not the CUDA path)

Usage:
    # Run on a machine with NPU hardware + torch_npu installed:
    python -m pytest test/float8/test_float8_npu_training.py -v
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from torchao.float8.config import Float8LinearConfig, e4m3_dtype
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_ops import (
    _addmm_float8_unwrapped_npu,
    addmm_float8_unwrapped,
)
from torchao.float8.float8_training_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import tensor_to_scale


def _is_npu_available():
    """Check if NPU hardware and torch_npu are available."""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def _make_linear_mm_config():
    """Create a default LinearMMConfig for testing."""
    default_scaled_mm_config = ScaledMMConfig()
    return LinearMMConfig(
        default_scaled_mm_config,
        default_scaled_mm_config,
        default_scaled_mm_config,
    )


def _randn_fp8(shape, device="npu"):
    """Create a random float8_e4m3fn tensor by generating in bfloat16 first, then casting.

    torch.randn does not support float8 dtypes directly because the normal
    distribution kernel is not implemented for Float8_e4m3fn on any device.
    """
    return torch.randn(shape, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )


# ============================================================================
# All tests require NPU hardware since float8 tensors must live on device
# ============================================================================


@unittest.skipUnless(_is_npu_available(), "NPU hardware not available")
class TestNPUDispatch(unittest.TestCase):
    """Tests that verify the NPU dispatch path is taken on real NPU tensors."""

    def test_addmm_float8_unwrapped_dispatches_to_npu(self):
        """Verify addmm_float8_unwrapped calls _addmm_float8_unwrapped_npu for NPU tensors."""
        M, K, N = 32, 64, 48

        # Create fp8 tensors on NPU: randn in bf16 then cast
        a_data = _randn_fp8((M, K))
        b_data = _randn_fp8((K, N))
        a_scale = torch.tensor(1.0, device="npu")
        b_scale = torch.tensor(1.0, device="npu")

        with patch(
            "torchao.float8.float8_ops._addmm_float8_unwrapped_npu",
            wraps=_addmm_float8_unwrapped_npu,
        ) as spy:
            result = addmm_float8_unwrapped(
                a_data,
                a_scale,
                b_data,
                b_scale,
                output_dtype=torch.bfloat16,
            )
            # Assert the NPU path was called
            spy.assert_called_once()
            call_args = spy.call_args[0]
            # _addmm_float8_unwrapped_npu(a_data, a_scale, b_data, b_scale, output_dtype)
            assert call_args[0] is a_data  # a_data (x1)
            assert call_args[1] is a_scale  # a_scale (pertoken_scale)
            assert call_args[2] is b_data  # b_data (x2)
            assert call_args[3] is b_scale  # b_scale (scale)
            assert call_args[4] == torch.bfloat16  # output_dtype

    def test_npu_path_passes_direct_scale_not_inverse(self):
        """Verify NPU path passes scale directly (not reciprocal).

        This is important because npu_quant_matmul expects direct scales
        (fp8_val * scale = hp_val), unlike torch._scaled_mm which expects
        inverse scales.
        """
        M, K, N = 32, 64, 48
        a_data = _randn_fp8((M, K))
        b_data = _randn_fp8((K, N))
        a_scale = torch.tensor(2.5, device="npu")
        b_scale = torch.tensor(3.7, device="npu")

        with patch(
            "torchao.float8.float8_ops._addmm_float8_unwrapped_npu",
            wraps=_addmm_float8_unwrapped_npu,
        ) as spy:
            addmm_float8_unwrapped(
                a_data,
                a_scale,
                b_data,
                b_scale,
                output_dtype=torch.bfloat16,
            )
            call_args = spy.call_args[0]
            # Scales must be passed directly, not as reciprocals
            # a_scale → pertoken_scale, b_scale → scale in npu_quant_matmul
            assert torch.equal(call_args[1], a_scale), (
                f"Expected direct a_scale={a_scale}, got {call_args[1]}"
            )
            assert torch.equal(call_args[3], b_scale), (
                f"Expected direct b_scale={b_scale}, got {call_args[3]}"
            )

    def test_npu_path_with_bias(self):
        """Verify bias is forwarded to the NPU path."""
        M, K, N = 32, 64, 48
        a_data = _randn_fp8((M, K))
        b_data = _randn_fp8((K, N))
        a_scale = torch.tensor(1.0, device="npu")
        b_scale = torch.tensor(1.0, device="npu")
        bias = torch.randn(N, dtype=torch.bfloat16, device="npu")

        with patch(
            "torchao.float8.float8_ops._addmm_float8_unwrapped_npu",
            wraps=_addmm_float8_unwrapped_npu,
        ) as spy:
            addmm_float8_unwrapped(
                a_data,
                a_scale,
                b_data,
                b_scale,
                output_dtype=torch.bfloat16,
                bias=bias,
            )
            call_kwargs = spy.call_args[1]
            assert call_kwargs["bias"] is bias

    def test_float8_mm_dispatch_uses_npu_path(self):
        """Verify that Float8TrainingTensor mm dispatch goes through NPU addmm path."""
        M, K, N = 32, 64, 48
        linear_mm_config = _make_linear_mm_config()

        # Create hp tensors on NPU and convert to Float8TrainingTensor
        a_hp = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
        b_hp = torch.randn(K, N, dtype=torch.bfloat16, device="npu")

        a_scale = tensor_to_scale(a_hp, e4m3_dtype)
        b_scale = tensor_to_scale(b_hp, e4m3_dtype)

        a_fp8 = hp_tensor_and_scale_to_float8(
            a_hp, a_scale, e4m3_dtype, linear_mm_config, GemmInputRole.INPUT
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b_hp, b_scale, e4m3_dtype, linear_mm_config, GemmInputRole.WEIGHT
        )

        # Spy on the NPU function to verify it's called
        with patch(
            "torchao.float8.float8_ops._addmm_float8_unwrapped_npu",
            wraps=_addmm_float8_unwrapped_npu,
        ) as spy:
            result = torch.mm(a_fp8, b_fp8)
            spy.assert_called_once()
            # Verify output shape and dtype
            self.assertEqual(result.shape, (M, N))
            self.assertEqual(result.dtype, torch.bfloat16)


# ============================================================================
# Hardware tests: end-to-end training on NPU
# ============================================================================


@unittest.skipUnless(_is_npu_available(), "NPU hardware not available")
class TestNPUFloat8TrainingHardware(unittest.TestCase):
    """Tests that run on actual NPU hardware."""

    def test_addmm_float8_unwrapped_npu_basic(self):
        """Test _addmm_float8_unwrapped_npu produces correct results."""
        M, K, N = 64, 128, 96

        # Create hp reference
        a_hp = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
        b_hp = torch.randn(K, N, dtype=torch.bfloat16, device="npu")

        # Reference: hp matmul
        ref_output = torch.mm(a_hp, b_hp)

        # FP8 path
        a_scale = tensor_to_scale(a_hp, e4m3_dtype)
        b_scale = tensor_to_scale(b_hp, e4m3_dtype)

        a_fp8_data = (a_hp / a_scale).to(torch.float8_e4m3fn)
        b_fp8_data = (b_hp / b_scale).to(torch.float8_e4m3fn)

        fp8_output = _addmm_float8_unwrapped_npu(
            a_fp8_data,
            a_scale,
            b_fp8_data,
            b_scale,
            output_dtype=torch.bfloat16,
        )

        # Check shapes match
        self.assertEqual(fp8_output.shape, ref_output.shape)
        self.assertEqual(fp8_output.dtype, torch.bfloat16)

        # Check numerical closeness (FP8 has limited precision)
        # SQNR > 15 dB is a reasonable threshold for FP8
        from torchao.float8.float8_utils import compute_error

        sqnr = compute_error(ref_output, fp8_output)
        self.assertGreater(
            sqnr, 15.0, f"SQNR too low: {sqnr} dB (expected > 15 dB)"
        )

    def test_addmm_float8_unwrapped_npu_with_bias(self):
        """Test NPU fp8 matmul with bias."""
        M, K, N = 64, 128, 96

        a_hp = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
        b_hp = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
        bias = torch.randn(N, dtype=torch.bfloat16, device="npu")

        ref_output = torch.mm(a_hp, b_hp) + bias

        a_scale = tensor_to_scale(a_hp, e4m3_dtype)
        b_scale = tensor_to_scale(b_hp, e4m3_dtype)
        a_fp8_data = (a_hp / a_scale).to(torch.float8_e4m3fn)
        b_fp8_data = (b_hp / b_scale).to(torch.float8_e4m3fn)

        fp8_output = _addmm_float8_unwrapped_npu(
            a_fp8_data,
            a_scale,
            b_fp8_data,
            b_scale,
            output_dtype=torch.bfloat16,
            bias=bias,
        )

        self.assertEqual(fp8_output.shape, ref_output.shape)

    def test_float8_linear_forward_npu(self):
        """Test Float8Linear forward pass on NPU."""
        M, K, N = 64, 128, 96

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        linear = nn.Linear(K, N, bias=False, device="npu", dtype=torch.bfloat16)
        fp8_linear = Float8Linear.from_float(linear, config=config)

        x = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
        output = fp8_linear(x)

        self.assertEqual(output.shape, (M, N))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_float8_linear_backward_npu(self):
        """Test Float8Linear backward pass on NPU (gradients flow through NPU path)."""
        M, K, N = 64, 128, 96

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        linear = nn.Linear(K, N, bias=False, device="npu", dtype=torch.bfloat16)
        fp8_linear = Float8Linear.from_float(linear, config=config)

        x = torch.randn(M, K, device="npu", dtype=torch.bfloat16, requires_grad=True)
        output = fp8_linear(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (M, K))
        self.assertIsNotNone(fp8_linear.weight.grad)
        self.assertEqual(fp8_linear.weight.grad.shape, (N, K))

    def test_float8_linear_with_bias_npu(self):
        """Test Float8Linear with bias on NPU."""
        M, K, N = 64, 128, 96

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        linear = nn.Linear(K, N, bias=True, device="npu", dtype=torch.bfloat16)
        fp8_linear = Float8Linear.from_float(linear, config=config)

        x = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
        output = fp8_linear(x)

        self.assertEqual(output.shape, (M, N))

    def test_convert_to_float8_training_npu(self):
        """Test full convert_to_float8_training pipeline on NPU."""
        model = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
        ).bfloat16().npu()

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config)

        # Verify conversion happened
        self.assertIsInstance(model[0], Float8Linear)
        self.assertIsInstance(model[2], Float8Linear)

        # Forward
        x = torch.randn(16, 128, device="npu", dtype=torch.bfloat16)
        output = model(x)
        self.assertEqual(output.shape, (16, 32))

        # Backward
        output.sum().backward()
        self.assertIsNotNone(model[0].weight.grad)
        self.assertIsNotNone(model[2].weight.grad)

    def test_training_loop_npu(self):
        """Test a full training loop on NPU to verify convergence."""
        torch.manual_seed(42)
        M, K, N = 64, 128, 32

        model = nn.Sequential(
            nn.Linear(K, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, N, bias=False),
        ).bfloat16().npu()

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(M, K, device="npu", dtype=torch.bfloat16)
        target = torch.randn(M, N, device="npu", dtype=torch.bfloat16)

        initial_loss = None
        final_loss = None
        for step in range(20):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Loss should decrease over training
        self.assertLess(
            final_loss,
            initial_loss,
            f"Training did not converge: initial_loss={initial_loss}, final_loss={final_loss}",
        )

    def test_npu_path_actually_called_in_training(self):
        """Verify npu_quant_matmul is actually called during fwd+bwd, not CUDA _scaled_mm."""
        M, K, N = 64, 128, 96

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        linear = nn.Linear(K, N, bias=False, device="npu", dtype=torch.bfloat16)
        fp8_linear = Float8Linear.from_float(linear, config=config)

        x = torch.randn(M, K, device="npu", dtype=torch.bfloat16)

        with patch(
            "torchao.float8.float8_ops._addmm_float8_unwrapped_npu",
            wraps=_addmm_float8_unwrapped_npu,
        ) as spy:
            output = fp8_linear(x)
            output.sum().backward()

            # Forward: 1 call (output = input @ weight)
            # Backward: 2 calls (grad_input = grad_output @ weight,
            #                     grad_weight = grad_output.T @ input)
            self.assertEqual(
                spy.call_count,
                3,
                f"Expected 3 calls to NPU path (1 fwd + 2 bwd), got {spy.call_count}",
            )


# ============================================================================
# Realistic user-facing test: how a real user would use torchao FP8 training
# ============================================================================


class SimpleMLP(nn.Module):
    """A small MLP resembling a transformer feed-forward block."""

    def __init__(self, d_model, d_ff, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = d_model
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, d_ff))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_ff
        layers.append(nn.Linear(in_dim, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@unittest.skipUnless(_is_npu_available(), "NPU hardware not available")
class TestNPUFloat8TrainingUserWorkflow(unittest.TestCase):
    """
    Realistic tests that mirror how a user would actually use torchao
    to train a model with FP8 on NPU.

    Follows the documented API:
        1. Create model
        2. Pick a recipe via Float8LinearConfig.from_recipe_name(...)
        3. convert_to_float8_training(model, config=config)
        4. Train normally with standard optimizer and loss
    """

    def _train_loop(self, model, train_x, train_y, steps=50, lr=1e-2):
        """Run a standard training loop and return (initial_loss, final_loss)."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        initial_loss = None
        final_loss = None
        for step in range(steps):
            optimizer.zero_grad()
            output = model(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
        return initial_loss, final_loss

    def test_simple_mlp_fp8_training(self):
        """
        A user creates a 2-layer MLP on NPU, converts to FP8 training with
        the tensorwise recipe, and trains on a regression task.

        Equivalent user code:
            model = SimpleMLP(256, 512).bfloat16().npu()
            config = Float8LinearConfig.from_recipe_name("tensorwise")
            convert_to_float8_training(model, config=config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
            for batch in dataloader:
                ...
        """
        torch.manual_seed(42)
        d_model, d_ff, batch_size = 256, 512, 64

        # 1. Create model
        model = SimpleMLP(d_model, d_ff, num_layers=2).bfloat16().npu()

        # 2. Pick a recipe
        config = Float8LinearConfig.from_recipe_name("tensorwise")

        # 3. Convert to FP8 training
        convert_to_float8_training(model, config=config)

        # Verify all nn.Linear layers were swapped to Float8Linear
        linear_modules = [
            m for m in model.modules() if isinstance(m, (nn.Linear, Float8Linear))
        ]
        for m in linear_modules:
            self.assertIsInstance(m, Float8Linear, f"Expected Float8Linear, got {type(m)}")

        # 4. Train on a simple regression task
        train_x = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)
        train_y = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)

        initial_loss, final_loss = self._train_loop(model, train_x, train_y, steps=50)

        # Loss must decrease — training actually works
        self.assertLess(
            final_loss,
            initial_loss * 0.5,
            f"Training did not converge sufficiently: "
            f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}",
        )

    def test_multi_layer_model_fp8_training(self):
        """
        A deeper model (4 linear layers) to verify FP8 training works with
        multiple stacked Float8Linear modules.
        """
        torch.manual_seed(123)
        d_model, d_ff, batch_size = 128, 256, 32

        model = SimpleMLP(d_model, d_ff, num_layers=4).bfloat16().npu()
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config)

        train_x = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)
        train_y = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)

        initial_loss, final_loss = self._train_loop(model, train_x, train_y, steps=80)

        self.assertLess(
            final_loss,
            initial_loss * 0.5,
            f"Deep model did not converge: "
            f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}",
        )

    def test_fp8_training_with_bias(self):
        """
        Verify FP8 training works with biased linear layers, which is the
        default for nn.Linear.
        """
        torch.manual_seed(77)
        batch_size = 32

        # Users often don't explicitly set bias=False
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        ).bfloat16().npu()

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config)

        train_x = torch.randn(batch_size, 128, device="npu", dtype=torch.bfloat16)
        train_y = torch.randn(batch_size, 64, device="npu", dtype=torch.bfloat16)

        initial_loss, final_loss = self._train_loop(model, train_x, train_y, steps=50)

        self.assertLess(
            final_loss,
            initial_loss * 0.5,
            f"Biased model did not converge: "
            f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}",
        )

    def test_fp8_vs_bf16_accuracy(self):
        """
        Compare FP8 training against BF16 baseline to ensure FP8 does not
        degrade accuracy significantly on a simple task.
        """
        torch.manual_seed(99)
        d_model, d_ff, batch_size = 128, 256, 32

        train_x = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)
        train_y = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)

        # BF16 baseline
        torch.manual_seed(99)
        bf16_model = SimpleMLP(d_model, d_ff).bfloat16().npu()
        _, bf16_loss = self._train_loop(bf16_model, train_x, train_y, steps=60)

        # FP8 model (same init)
        torch.manual_seed(99)
        fp8_model = SimpleMLP(d_model, d_ff).bfloat16().npu()
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(fp8_model, config=config)
        _, fp8_loss = self._train_loop(fp8_model, train_x, train_y, steps=60)

        # FP8 loss should be in the same ballpark as BF16 (within 3x)
        self.assertLess(
            fp8_loss,
            bf16_loss * 3.0,
            f"FP8 loss ({fp8_loss:.4f}) is much worse than BF16 loss ({bf16_loss:.4f})",
        )

    def test_fp8_training_npu_path_verified(self):
        """
        Full user workflow with verification that npu_quant_matmul is actually
        called (not falling back to some other path).
        """
        torch.manual_seed(42)
        d_model, batch_size = 128, 16

        model = nn.Sequential(
            nn.Linear(d_model, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, d_model, bias=False),
        ).bfloat16().npu()

        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        train_x = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)
        train_y = torch.randn(batch_size, d_model, device="npu", dtype=torch.bfloat16)

        with patch(
            "torchao.float8.float8_ops._addmm_float8_unwrapped_npu",
            wraps=_addmm_float8_unwrapped_npu,
        ) as spy:
            # Run 5 training steps
            for _ in range(5):
                optimizer.zero_grad()
                output = model(train_x)
                loss = nn.functional.mse_loss(output, train_y)
                loss.backward()
                optimizer.step()

            # 2 Float8Linear layers × (1 fwd + 2 bwd) = 6 calls per step
            # 5 steps × 6 = 30 total calls
            self.assertEqual(
                spy.call_count,
                30,
                f"Expected 30 NPU calls (5 steps × 2 layers × 3 GEMMs), "
                f"got {spy.call_count}",
            )

    def test_module_filter_fn(self):
        """
        Verify that module_filter_fn lets users selectively convert only
        certain layers to FP8, leaving others in BF16.
        """
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(128, 256, bias=False),  # index 0: convert
            nn.ReLU(),
            nn.Linear(256, 64, bias=False),   # index 2: keep bf16
        ).bfloat16().npu()

        config = Float8LinearConfig.from_recipe_name("tensorwise")

        # Only convert the first linear layer
        def filter_fn(mod, fqn):
            return "0" in fqn

        convert_to_float8_training(model, config=config, module_filter_fn=filter_fn)

        self.assertIsInstance(model[0], Float8Linear)
        self.assertNotIsInstance(model[2], Float8Linear)
        self.assertIsInstance(model[2], nn.Linear)

        # Should still train fine with mixed FP8 + BF16 layers
        train_x = torch.randn(16, 128, device="npu", dtype=torch.bfloat16)
        output = model(train_x)
        output.sum().backward()

        self.assertIsNotNone(model[0].weight.grad)
        self.assertIsNotNone(model[2].weight.grad)


if __name__ == "__main__":
    unittest.main()
