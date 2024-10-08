import importlib
import pytest
import torch
from torchao._models.llama.model import Transformer
from torchao.prototype.spinquant.spinquant import apply_spinquant


def _is_package_available(pkg_name):
    return importlib.util.find_spec(pkg_name) is not None


def _init_model(name="7B", device="cpu", precision=torch.bfloat16):
    model = Transformer.from_name(name)
    model.to(device=device, dtype=precision)
    return model.eval()


_AVAILABLE_DEVICES = ["cpu"]
if torch.cuda.is_available() and _is_package_available("fast_hadamard_transform"):
    _AVAILABLE_DEVICES.append("cuda")


@pytest.mark.parametrize("device", _AVAILABLE_DEVICES)
def test_spinquant_no_quantization(device):
    model = _init_model(device=device)
    seq_len = 16
    batch_size = 1
    is_training = False
    input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(device)
    input_pos = None if is_training else torch.arange(seq_len).to(device)
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=seq_len, training=is_training)

    with torch.no_grad():
        out = model(input_ids, input_pos)
        apply_spinquant(model)
        out_spinquant = model(input_ids, input_pos)

    # Output should be the same without quantization (the rotations cancel out)
    # TODO: not sure if these atol/rtol are excessively large (it fails for smaller values)
    torch.testing.assert_close(out, out_spinquant, atol=5e-2, rtol=1e-2)


# TODO: test GPTQ compatability?