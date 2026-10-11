# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)

# (observer factory, whether it needs a 2D input for the channel axis)
_PER_TENSOR_OBSERVERS = [
    ("MinMaxObserver", MinMaxObserver),
    ("MovingAverageMinMaxObserver", MovingAverageMinMaxObserver),
]
_PER_CHANNEL_OBSERVERS = [
    ("PerChannelMinMaxObserver", PerChannelMinMaxObserver),
    ("MovingAveragePerChannelMinMaxObserver", MovingAveragePerChannelMinMaxObserver),
]


class TestObserverComplexInput(TestCase):
    """Observers must reject complex input rather than observing only its real part.

    These observers derive qparams from a real-valued min/max. Casting a complex
    tensor to the buffer dtype used to drop the imaginary part and return qparams
    that looked ordinary, while the quantize kernels reject non-float input a step
    later (ATen Quantizer.cpp).
    """

    def _complex(self, shape, dtype):
        real = torch.arange(1, 1 + torch.Size(shape).numel(), dtype=torch.float32)
        return torch.complex(real, real * 100).reshape(shape).to(dtype)

    def test_per_tensor_observers_reject_complex(self):
        for dtype in (torch.complex64, torch.complex128):
            for name, factory in _PER_TENSOR_OBSERVERS:
                x = self._complex((4,), dtype)
                with self.assertRaisesRegex(NotImplementedError, name):
                    factory()(x)
                # the message names the dtype, so a report says which input was passed
                with self.assertRaisesRegex(NotImplementedError, str(dtype)):
                    factory()(x)

    def test_per_channel_observers_reject_complex(self):
        for dtype in (torch.complex64, torch.complex128):
            for name, factory in _PER_CHANNEL_OBSERVERS:
                x = self._complex((2, 3), dtype)
                with self.assertRaisesRegex(NotImplementedError, name):
                    factory()(x)
                with self.assertRaisesRegex(NotImplementedError, str(dtype)):
                    factory()(x)

    def test_histogram_observer_rejects_complex(self):
        # HistogramObserver already refuses complex input, by way of torch.aminmax.
        # Pinned here so the five observers stay consistent with each other.
        with self.assertRaises(NotImplementedError):
            HistogramObserver()(self._complex((4,), torch.complex64))

    def test_real_input_is_unaffected(self):
        for _, factory in _PER_TENSOR_OBSERVERS:
            obs = factory()
            obs(torch.tensor([1.0, 2.0, 3.0]))
            scale, zero_point = obs.calculate_qparams()
            self.assertEqual(scale.numel(), 1)
            self.assertTrue(torch.all(scale > 0))
            self.assertFalse(zero_point.dtype.is_floating_point)
        for _, factory in _PER_CHANNEL_OBSERVERS:
            obs = factory()
            obs(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            scale, zero_point = obs.calculate_qparams()
            self.assertEqual(scale.numel(), 2)
            self.assertTrue(torch.all(scale > 0))

    def test_empty_complex_input_still_passes_through(self):
        # The numel() == 0 early return predates this check and is left alone:
        # an empty tensor is returned unobserved whatever its dtype.
        empty = torch.tensor([], dtype=torch.complex64)
        out = MinMaxObserver()(empty)
        self.assertEqual(out.numel(), 0)
        self.assertEqual(out.dtype, torch.complex64)


if __name__ == "__main__":
    run_tests()
