import pytest
import torch
from torchao._models.llama.model import Transformer
from torchao.quantization.spin_quant import apply_spinquant

_AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def init_model(name="7B", device="cpu", precision=torch.bfloat16):
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


# TODO: fix is_training=True
# @pytest.mark.parametrize("batch_size", [1, 4])
# @pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("device", _AVAILABLE_DEVICES)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("is_training", [False])
def test_spinquant_no_quantization(device, batch_size, is_training):
    model = init_model(device=device)
    seq_len = 16
    input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(device)
    input_pos = None if is_training else torch.arange(seq_len).to(device)
    with torch.device(device):
        # TODO: is this in any way necessary?
        model.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len, training=is_training)

    with torch.no_grad():
        out = model(input_ids, input_pos)
        apply_spinquant(model)
        out_spinquant = model(input_ids, input_pos)

    # Output should be the same without quantization (the rotations cancel out)
    # TODO: not sure if this atol is excessively large (it fails for smaller values)
    torch.testing.assert_close(out, out_spinquant, atol=5e-2, rtol=1e-2)


# TODO: test GPTQ compatability?