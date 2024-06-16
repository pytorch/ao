import pytest
import torch
import torch.nn as nn
from torchao.prototype.dtypes import BitnetTensor
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter

def _apply_weight_only_uint2_quant(model):
    def fn(mod):
        mod.weight = torch.nn.Parameter(BitnetTensor.from_float(mod.weight), requires_grad=False)
        return mod

    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: fn(mod),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )

@pytest.mark.parametrize("input_shape", [[2, 4], [5, 5, 5, 4], [1, 4, 4]])
def test_uint2_quant(input_shape):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(*input_shape).to(device)
    m = nn.Sequential(nn.Linear(4, 16)).to(device)
    y_ref = m(x)
    _apply_weight_only_uint2_quant(m)
    y_wo = m(x)
    assert y_ref.shape == y_wo.shape
    # WIP - Need to use the latest build and test torch.compile
    # y_compiled = torch.compile(m, fullgraph=True)(x)

if __name__ == '__main__':
    test_uint2_quant([2, 4])
