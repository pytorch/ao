# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4089.

`_is_128_128_scaled` used to infer "128x128 blockwise scaling" purely from
`block_size == (128, 128)`. That's ambiguous: a `PerTensor`-granularity
weight whose shape happens to be exactly 128x128 also gets `block_size ==
(128, 128)` (one block spanning the whole tensor, from `get_block_size`),
even though it was never blockwise-scaled. Downstream dispatch code
(`torchao/quantization/quantize_/workflows/float8/float8_tensor.py`) uses
`_is_128_128_scaled` to decide whether the *activation* tensor must be 1x128
scaled, which then fails for a plain PerTensor-quantized activation - this is
exactly the crash the issue reports for `ToyLinearModel`'s 128x128 linear.

These tests exercise `_is_128_128_scaled`/`_is_tensorwise_scaled` directly
with plain CPU tensors carrying a `block_size` attribute (the only thing
these helpers actually touch - no float8 dtype or GPU compute is involved),
so they run happily in CPU-only CI.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.float8.inference import _is_128_128_scaled, _is_tensorwise_scaled


def _with_block_size(shape, block_size):
    t = torch.empty(shape)
    t.block_size = block_size
    return t


class TestIs128128Scaled(TestCase):
    def test_pertensor_128x128_is_not_128x128_scaled(self):
        # PerTensor granularity on an exactly-128x128 tensor: get_block_size()
        # returns the full tensor shape, i.e. (128, 128) - same numbers as
        # genuine 128x128 blockwise scaling, but semantically PerTensor.
        x = _with_block_size((128, 128), [128, 128])
        self.assertTrue(_is_tensorwise_scaled(x))
        self.assertFalse(_is_128_128_scaled(x))

    def test_genuine_128x128_blockwise_on_larger_tensor_is_detected(self):
        # A 256x256 tensor tiled into 128x128 blocks is unambiguous: block_size
        # (128, 128) does *not* equal the full (256, 256) shape, so it can only
        # mean blockwise scaling.
        x = _with_block_size((256, 256), [128, 128])
        self.assertFalse(_is_tensorwise_scaled(x))
        self.assertTrue(_is_128_128_scaled(x))

    def test_non_128_128_block_size_is_not_128x128_scaled(self):
        x = _with_block_size((256, 256), [1, 256])  # rowwise
        self.assertFalse(_is_128_128_scaled(x))

    def test_pertensor_non_128_shape_is_not_128x128_scaled(self):
        # Sanity check: PerTensor on a shape that isn't 128x128 was already
        # correctly excluded before this fix (included for completeness).
        x = _with_block_size((64, 64), [64, 64])
        self.assertTrue(_is_tensorwise_scaled(x))
        self.assertFalse(_is_128_128_scaled(x))


if __name__ == "__main__":
    run_tests()
