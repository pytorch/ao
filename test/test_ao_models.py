# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch

from torchao._models.llama.model import Transformer
from torchao.testing import common_utils

_AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_BATCH_SIZES = [1, 4]
_TRAINING_MODES = [True, False]

# Define test parameters
COMMON_DEVICES = common_utils.parametrize("device", _AVAILABLE_DEVICES)
COMMON_DTYPES = common_utils.parametrize("dtype", [torch.float32, torch.bfloat16])


def init_model(name="stories15M", device="cpu", precision=torch.bfloat16):
    """Initialize and return a Transformer model with specified configuration."""
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


class TorchAOBasicTestCase(unittest.TestCase):
    """Test suite for basic Transformer inference functionality."""

    @COMMON_DEVICES
    @common_utils.parametrize("batch_size", _BATCH_SIZES)
    @common_utils.parametrize("is_training", _TRAINING_MODES)
    def test_ao_inference_mode(self, device, batch_size, is_training):
        # Initialize model with specified device
        random_model = init_model(device=device)

        # Set up test input parameters
        seq_len = 16
        input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(device)

        # input_pos is None for training mode, tensor for inference mode
        input_pos = None if is_training else torch.arange(seq_len).to(device)

        # Setup model caches within the device context
        with torch.device(device):
            random_model.setup_caches(
                max_batch_size=batch_size, max_seq_length=seq_len, training=is_training
            )

        # Run multiple inference iterations to ensure consistency
        for i in range(3):
            out = random_model(input_ids, input_pos)
            self.assertIsNotNone(out, f"Model failed to run on iteration {i}")


common_utils.instantiate_parametrized_tests(TorchAOBasicTestCase)

if __name__ == "__main__":
    unittest.main()
