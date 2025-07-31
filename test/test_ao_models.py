# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch

from torchao._models.llama.model import Transformer

_AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def init_model(name="stories15M", device="cpu", precision=torch.bfloat16):
    """Initialize and return a Transformer model with specified configuration."""
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


class TestAOLlamaModel(unittest.TestCase):
    """Test suite for AO Llama model inference functionality."""

    def test_ao_llama_model_inference_mode(self):
        """Test model inference across different devices, batch sizes, and training modes."""
        # Define test parameters
        devices = _AVAILABLE_DEVICES
        batch_sizes = [1, 4]
        training_modes = [True, False]

        # Iterate through all parameter combinations
        for device in devices:
            for batch_size in batch_sizes:
                for is_training in training_modes:
                    # Use subTest to create individual test cases for each parameter combination
                    with self.subTest(
                        device=device, batch_size=batch_size, is_training=is_training
                    ):
                        self._test_ao_llama_model_inference_mode(
                            device, batch_size, is_training
                        )

    def _test_ao_llama_model_inference_mode(self, device, batch_size, is_training):
        """Helper method to run a single inference test with given parameters."""
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


if __name__ == "__main__":
    unittest.main()
