# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchao._models.llama.model import Transformer
from torchao.utils import get_available_devices


_DEVICES = get_available_devices()


def init_model(name="stories15M", device="cpu", precision=torch.bfloat16):
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("is_training", [True, False])
def test_ao_llama_model_inference_mode(device, batch_size, is_training):
    random_model = init_model(device=device)
    seq_len = 16
    input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(device)
    input_pos = None if is_training else torch.arange(seq_len).to(device)
    with torch.device(device):
        random_model.setup_caches(
            max_batch_size=batch_size, max_seq_length=seq_len, training=is_training
        )
    for i in range(3):
        out = random_model(input_ids, input_pos)
        assert out is not None, "model failed to run"
