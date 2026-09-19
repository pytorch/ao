# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase

from torchao.quantization.pt2e.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)

_MIN_MAX_OBSERVERS = [
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
]


class TestObserverComplexInput(TestCase):
    """The min/max observers used to accept complex input.

    ``x.to(self.min_val.dtype)`` dropped the imaginary part, so the qparams were
    calibrated on the real part alone, while the quantize kernels reject the same
    tensor with "Quantize only works on Float Tensor". ``HistogramObserver``
    already raised, through ``torch.aminmax``.
    """

    @common_utils.parametrize("observer_cls", _MIN_MAX_OBSERVERS)
    @common_utils.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_min_max_observers_reject_complex_input(self, observer_cls, dtype):
        x = torch.tensor([[1 + 100j, 2 + 200j], [3 + 300j, 4 + 400j]], dtype=dtype)
        observer = observer_cls()

        with self.assertRaisesRegex(
            NotImplementedError,
            f"{observer_cls.__name__} does not support complex input, got {dtype}",
        ):
            observer(x)

    def test_histogram_observer_still_rejects_complex_input(self):
        # HistogramObserver rejects complex through torch.aminmax rather than
        # through the check added here, so the exception is torch's and its type
        # has changed: the dtype dispatch raised NotImplementedError through 2.14,
        # and TORCH_CHECK_TYPE in ReduceOps.cpp raises TypeError on current main.
        # Only assert that it does not silently accept.
        observer = HistogramObserver()

        with self.assertRaises((NotImplementedError, TypeError)):
            observer(torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64))

    @common_utils.parametrize("observer_cls", _MIN_MAX_OBSERVERS)
    def test_real_input_is_unaffected(self, observer_cls):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        observer = observer_cls()

        observer(x)
        scale, zero_point = observer.calculate_qparams()

        self.assertTrue(torch.all(scale > 0))
        self.assertEqual(scale.numel(), zero_point.numel())

    @common_utils.parametrize("observer_cls", _MIN_MAX_OBSERVERS)
    def test_empty_input_still_short_circuits(self, observer_cls):
        observer = observer_cls()
        empty = torch.empty(0)

        self.assertEqual(observer(empty).numel(), 0)


class TestObserverTorchScript(TestCase):
    """These observers are scriptable and must stay that way.

    ``observer.py`` notes that "Observers must be torchscriptable however", and
    the module is decorated with ``@torch.jit.export`` throughout. The complex
    check is easy to write in a way that silently breaks this: ``type(self)``
    inside ``forward`` resolves to ``Tensor.type`` under TorchScript, so
    ``type(self).__name__`` fails to compile while every eager test keeps
    passing.
    """

    @common_utils.parametrize("observer_cls", _MIN_MAX_OBSERVERS)
    def test_observer_is_scriptable(self, observer_cls):
        scripted = torch.jit.script(observer_cls())

        scripted(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        scale, zero_point = scripted.calculate_qparams()

        self.assertTrue(torch.all(scale > 0))
        self.assertEqual(scale.numel(), zero_point.numel())

    @common_utils.parametrize("observer_cls", _MIN_MAX_OBSERVERS)
    def test_scripted_observer_rejects_complex_input(self, observer_cls):
        scripted = torch.jit.script(observer_cls())
        x = torch.tensor([[1 + 100j, 2 + 200j]], dtype=torch.complex64)

        with self.assertRaisesRegex(
            torch.jit.Error,
            f"{observer_cls.__name__} does not support complex input",
        ):
            scripted(x)


common_utils.instantiate_parametrized_tests(TestObserverComplexInput)
common_utils.instantiate_parametrized_tests(TestObserverTorchScript)

if __name__ == "__main__":
    unittest.main()
