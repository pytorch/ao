# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]
"""Tests for FakeQuantize's sync-free CUDA path.

For CUDA inputs, FakeQuantize evaluates ``observer_enabled`` /
``fake_quant_enabled`` on the device instead of reading them on the host.

Naming used below:
  * "host path"   = the original forward (``_device_flags = False``)
  * "device path" = the new forward       (``_device_flags = True``, the default)

The device path must match the host path bit for bit (outputs, input
gradients, every state-dict tensor) and must not synchronize with the host.
"""

import copy
import unittest
import warnings

import torch

from torchao.quantization.pt2e import observer as O
from torchao.quantization.pt2e.fake_quantize import FakeQuantize

CUDA = "cuda"
SHAPE = (6, 8, 1, 5)

# One entry per observer type and qscheme branch of _calculate_qparams_device.
CONFIGS = {
    "mavg_per_tensor_affine": dict(
        observer=O.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    ),
    "mavg_per_tensor_symmetric": dict(
        observer=O.MovingAverageMinMaxObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    ),
    # Dynamic-activation QAT (XNNPACK / x86-inductor / Arm quantizers).
    "mavg_per_tensor_dynamic": dict(
        observer=O.MovingAverageMinMaxObserver,
        averaging_constant=1,
        is_dynamic=True,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    ),
    "minmax_per_tensor_affine_quint8": dict(
        observer=O.MinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
    # Exercises the quint8 fixed-zero-point branch of the symmetric qparams.
    "minmax_per_tensor_symmetric_quint8": dict(
        observer=O.MinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_symmetric,
    ),
    "per_channel_minmax_symmetric": dict(
        observer=O.PerChannelMinMaxObserver,
        quant_min=-127,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
    ),
    "per_channel_mavg_affine_axis1": dict(
        observer=O.MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_affine,
        ch_axis=1,
    ),
}
PER_CHANNEL_CONFIGS = ("per_channel_minmax_symmetric", "per_channel_mavg_affine_axis1")

# Flag changes applied before the forward at the given step, covering every way
# users flip the flags: the helpers, direct buffer writes and load_state_dict.
# Steps not listed run another forward with the current flags.
NUM_STEPS = 14
SNAPSHOT_STEP = 2  # state_dict saved here (both flags on) and restored at step 10
FLAG_CHANGES = {
    3: lambda m, snapshot: m.disable_observer(),
    5: lambda m, snapshot: m.enable_observer(),
    6: lambda m, snapshot: m.fake_quant_enabled.fill_(0),  # direct write
    8: lambda m, snapshot: m.fake_quant_enabled.fill_(1),  # direct write
    9: lambda m, snapshot: m.observer_enabled.fill_(0),  # direct write
    10: lambda m, snapshot: m.load_state_dict(snapshot),  # flags back to on/on
    12: lambda m, snapshot: (m.disable_fake_quant(), m.disable_observer()),
}

# fake_quantize_per_channel_affine itself still reads the zero-point range on
# the host inside ATen; removing those reads is pytorch/pytorch work.
MAX_PER_CHANNEL_ATEN_SYNCS = 2


def _as_bits(t):
    """Reinterpret floats as same-width ints so -0.0 != 0.0 and NaN == NaN."""
    t = t.detach().contiguous()
    int_view = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }.get(t.dtype)
    return t.view(int_view) if int_view is not None else t


def _host_and_device_pair(cfg):
    """Two identical FakeQuantize modules, one per path."""
    host = FakeQuantize(**cfg).to(CUDA)
    host._device_flags = False
    device = copy.deepcopy(host)
    device._device_flags = True
    return host, device


def _random_input(scale=1.0):
    x = torch.randn(SHAPE, device=CUDA) * scale
    # A signed zero in the input checks that gradients keep the sign of zero.
    x[0, 0, 0, 0] = -0.0
    return x


def _count_host_syncs(fn):
    """Number of CUDA synchronizations triggered while running ``fn``."""
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("synchronizing CUDA operation" in str(w.message) for w in caught)


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
class TestFakeQuantizeDeviceFlags(unittest.TestCase):
    def assertBitwiseEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertTrue(torch.equal(_as_bits(a), _as_bits(b)))

    def assertStepMatches(self, host, device, x):
        """One forward + backward on each module; everything must match bit for bit."""
        upstream_grad = torch.randn_like(x)
        x_host = x.clone().requires_grad_()
        x_device = x.clone().requires_grad_()
        y_host, y_device = host(x_host), device(x_device)
        (y_host * upstream_grad).sum().backward()
        (y_device * upstream_grad).sum().backward()

        self.assertBitwiseEqual(y_host, y_device)
        self.assertBitwiseEqual(x_host.grad, x_device.grad)
        state_host, state_device = host.state_dict(), device.state_dict()
        self.assertEqual(list(state_host), list(state_device))
        for key in state_host:
            self.assertBitwiseEqual(state_host[key], state_device[key])

    # --- Bitwise equality with the host path -------------------------------

    def test_matches_host_path_across_flag_changes(self):
        for name, cfg in CONFIGS.items():
            with self.subTest(name):
                torch.manual_seed(0)
                host, device = _host_and_device_pair(cfg)
                snapshot = None
                for step in range(NUM_STEPS):
                    if step == SNAPSHOT_STEP:
                        snapshot = copy.deepcopy(host.state_dict())
                    if step in FLAG_CHANGES:
                        FLAG_CHANGES[step](host, snapshot)
                        FLAG_CHANGES[step](device, snapshot)
                    x = _random_input(scale=step + 1)
                    if step > 0:
                        # Guard against a vacuous pass: the device path must
                        # actually run (per-channel sizes its buffers at step 0).
                        self.assertTrue(device._can_use_device_flags(x))
                    self.assertStepMatches(host, device, x)

    def test_matches_host_path_for_sized_but_unobserved_per_channel_buffers(self):
        # Per-channel buffers that are sized but hold the unobserved +inf / -inf.
        # Only a direct buffer write gets here: buffers start empty, and
        # reset_min_max_vals() / load_state_dict leave them empty or observed.
        #
        # Skipped: moving-average observer enabled. inf + c * (x - inf) = NaN
        # on both paths, and both then fail the min <= max check (the host path
        # with AssertionError, the device path with a device-side assert).
        for name in PER_CHANNEL_CONFIGS:
            moving_average = CONFIGS[name]["observer"] is not O.PerChannelMinMaxObserver
            for observer_on in (1, 0):
                for fake_quant_on in (1, 0):
                    if moving_average and observer_on:
                        continue
                    with self.subTest(
                        name, observer=observer_on, fake_quant=fake_quant_on
                    ):
                        torch.manual_seed(0)
                        host = FakeQuantize(**CONFIGS[name]).to(CUDA)
                        host._device_flags = False
                        host(torch.randn(SHAPE, device=CUDA))  # sizes the buffers
                        obs = host.activation_post_process
                        obs.min_val.fill_(float("inf"))
                        obs.max_val.fill_(float("-inf"))
                        host.observer_enabled.fill_(observer_on)
                        host.fake_quant_enabled.fill_(fake_quant_on)
                        device = copy.deepcopy(host)
                        device._device_flags = True

                        x = _random_input()
                        self.assertTrue(device._can_use_device_flags(x))
                        self.assertStepMatches(host, device, x)

    def test_qparams_check_matches_host_check(self):
        # _calculate_qparams_device returns the host path's min <= max check as
        # ``ok`` instead of asserting. Per-channel the check covers all channels,
        # as in check_min_max_valid, so one bad channel fails both. Valid
        # buffers give the host path's qparams bit for bit.
        for name, cfg in CONFIGS.items():
            obs = FakeQuantize(**cfg).to(CUDA).activation_post_process
            obs(torch.randn(SHAPE, device=CUDA))
            for bad in (None, "min_above_max", "nan"):
                with self.subTest(name, bad=bad):
                    min_val, max_val = obs.min_val.clone(), obs.max_val.clone()
                    first_min, first_max = min_val.view(-1), max_val.view(-1)
                    if bad == "min_above_max":
                        first_min[0] = first_max[0] + 1
                    elif bad == "nan":
                        first_min[0] = float("nan")
                    scale, zero_point, ok = obs._calculate_qparams_device(
                        min_val, max_val
                    )
                    self.assertEqual(ok.shape, ())
                    if bad is None:
                        self.assertTrue(ok)
                        host_scale, host_zero_point = obs._calculate_qparams(
                            min_val, max_val
                        )
                        self.assertBitwiseEqual(scale, host_scale)
                        self.assertBitwiseEqual(zero_point, host_zero_point)
                    else:
                        self.assertFalse(ok)
                        with self.assertRaises(AssertionError):
                            obs._calculate_qparams(min_val, max_val)

    # --- No host synchronization -------------------------------------------

    def test_forward_does_not_sync_with_host(self):
        for name, cfg in CONFIGS.items():
            with self.subTest(name):
                m = FakeQuantize(**cfg).to(CUDA)
                x = torch.randn(SHAPE, device=CUDA)
                m(x)  # per-channel buffers are sized on the first (host-path) call
                syncs = _count_host_syncs(lambda: m(x))
                if m.is_per_channel:
                    self.assertLessEqual(syncs, MAX_PER_CHANNEL_ATEN_SYNCS)
                else:
                    self.assertEqual(syncs, 0)

    # --- When the device path is (not) used ----------------------------------

    def test_unsupported_inputs_and_types_use_host_path(self):
        cfg = CONFIGS["mavg_per_tensor_symmetric"]

        class SubFakeQuantize(FakeQuantize):
            pass

        class SubObserver(O.MovingAverageMinMaxObserver):
            pass

        x = torch.randn(SHAPE, device=CUDA)
        supported = FakeQuantize(**cfg).to(CUDA)
        self.assertTrue(supported._can_use_device_flags(x))

        cases = {
            "cpu input": (FakeQuantize(**cfg), x.cpu()),
            "empty input": (supported, x[:0]),
            "FakeQuantize subclass": (SubFakeQuantize(**cfg).to(CUDA), x),
            "observer subclass": (
                FakeQuantize(**{**cfg, "observer": SubObserver}).to(CUDA),
                x,
            ),
            "per-channel before first call": (
                FakeQuantize(**CONFIGS["per_channel_minmax_symmetric"]).to(CUDA),
                x,
            ),
        }
        for case, (m, inp) in cases.items():
            with self.subTest(case):
                self.assertFalse(m._can_use_device_flags(inp))

    # --- torch.compile / TorchScript -----------------------------------------

    def test_compile_fullgraph_matches_eager(self):
        for name in ("mavg_per_tensor_symmetric", "per_channel_minmax_symmetric"):
            with self.subTest(name):
                torch._dynamo.reset()
                torch.manual_seed(0)
                m = FakeQuantize(**CONFIGS[name]).to(CUDA)
                m(torch.randn(SHAPE, device=CUDA))  # sizes per-channel buffers
                eager_ref = copy.deepcopy(m)
                eager_ref._device_flags = False
                compiled = torch.compile(m, fullgraph=True)  # raises on a graph break
                for step in range(3):
                    x = torch.randn(SHAPE, device=CUDA) * (step + 1)
                    x_compiled = x.clone().requires_grad_()
                    x_eager = x.clone().requires_grad_()
                    y_compiled, y_eager = compiled(x_compiled), eager_ref(x_eager)
                    y_compiled.sum().backward()
                    y_eager.sum().backward()
                    self.assertBitwiseEqual(y_compiled, y_eager)
                    self.assertBitwiseEqual(x_compiled.grad, x_eager.grad)
                self.assertBitwiseEqual(m.scale, eager_ref.scale)
                self.assertBitwiseEqual(m.zero_point, eager_ref.zero_point)

    def test_torchscript_compiles_host_path(self):
        # Scripted modules always take the host path; they must match eager.
        eager = FakeQuantize(**CONFIGS["mavg_per_tensor_affine"]).to(CUDA)
        eager._device_flags = False
        scripted = torch.jit.script(copy.deepcopy(eager))
        x = _random_input()
        self.assertBitwiseEqual(scripted(x), eager(x))
        self.assertBitwiseEqual(scripted.scale, eager.scale)
        self.assertBitwiseEqual(scripted.zero_point, eager.zero_point)


if __name__ == "__main__":
    unittest.main()
