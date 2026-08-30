# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import io
import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp

from torchao.prototype.moe_training.config import Float8TrainingOpConfig
from torchao.prototype.moe_training.tensor import (
    Float8TrainingWeightWrapperTensor,
)


def _make_wrapper(value: float) -> Float8TrainingWeightWrapperTensor:
    return Float8TrainingWeightWrapperTensor(
        torch.full((2, 2), value, dtype=torch.bfloat16),
        Float8TrainingOpConfig(),
    )


class Float8TrainingWeightWrapperSerializationTest(unittest.TestCase):
    def test_weights_only_round_trip(self) -> None:
        expected = _make_wrapper(1.0)
        serialized = io.BytesIO()
        torch.save(expected, serialized)

        serialized.seek(0)
        actual = torch.load(serialized, weights_only=True)

        self.assertIsInstance(actual, Float8TrainingWeightWrapperTensor)
        torch.testing.assert_close(actual._data, expected._data)
        self.assertEqual(expected.config, actual.config)

    def test_dcp_round_trip(self) -> None:
        expected = _make_wrapper(1.0)
        actual = _make_wrapper(0.0)

        with tempfile.TemporaryDirectory() as checkpoint_id:
            dcp.save({"weight": expected}, checkpoint_id=checkpoint_id)
            dcp.load({"weight": actual}, checkpoint_id=checkpoint_id)

        torch.testing.assert_close(actual._data, expected._data)
        self.assertEqual(expected.config, actual.config)
