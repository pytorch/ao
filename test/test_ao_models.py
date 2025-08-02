# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch

from torchao._models.llama.model import Transformer
from torchao.testing import common_utils


def init_model(name="stories15M", device="cpu", precision=torch.bfloat16):
    """Initialize and return a Transformer model with specified configuration."""
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


class TorchAOBasicTestCase(unittest.TestCase):
    """Test suite for basic Transformer inference functionality."""

    @common_utils.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    @common_utils.parametrize("batch_size", [1, 4])
    @common_utils.parametrize("is_training", [True, False])
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
