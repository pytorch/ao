import pytest
import torch
from torchao._models.llama.model import Transformer

_AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def init_model(name="stories15M", device="cpu", precision=torch.bfloat16):
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


@pytest.mark.parametrize("device", _AVAILABLE_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("has_input_pos", [True, False])
def test_ao_llama_model(device, batch_size, has_input_pos):
    random_model = init_model(device=device)
    seq_len = 16
    input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(device)
    input_pos = torch.arange(seq_len).to(device) if has_input_pos else None
    with torch.device(device):
        random_model.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len)
    out = random_model(input_ids, input_pos)
    assert out is not None, "model failed to run"
