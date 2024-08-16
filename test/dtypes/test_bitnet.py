import pytest
import torch
import torch.nn as nn
from torchao.prototype.dtypes import BitnetTensor
from torchao.prototype.dtypes.uint2 import unpack_uint2
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # source: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test  # noqa: E501

    # setup (currently do nothing)

    # tests will run here
    yield

    # teardown
    # avoid dynamo cache limit issues
    torch._dynamo.reset()

@pytest.fixture
def bitnet_tensor():
    input_tensor = torch.randint(0, 15, (4,4), dtype=torch.uint8)
    return BitnetTensor.from_unpacked(input_tensor)

def test_copy(bitnet_tensor):
    copied_tensor = bitnet_tensor.clone()
    assert torch.equal(bitnet_tensor.elem, copied_tensor.elem)

def test_transpose(bitnet_tensor):
    transposed_tensor = bitnet_tensor.t()
    expected_tensor = unpack_uint2(bitnet_tensor.elem).t()
    assert torch.equal(unpack_uint2(transposed_tensor.elem), expected_tensor)

def test_multiply(bitnet_tensor):
    w_t = torch.randint(0, 15, (4, 16), dtype=torch.uint8)
    w = BitnetTensor.from_unpacked(w_t)
    y = torch.addmm(torch.Tensor([1]), bitnet_tensor, w)

@pytest.mark.parametrize("dtype", [torch.float, torch.float16, torch.bfloat16, torch.int16, torch.int32, torch.int64])
def test_conversion(bitnet_tensor, dtype):
    converted_tensor = bitnet_tensor.to(dtype)
    expected_tensor = unpack_uint2(bitnet_tensor.elem).to(dtype)
    assert torch.allclose(converted_tensor, expected_tensor, atol=1e-5)

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
    y_compiled = torch.compile(m, fullgraph=True)(x)


if __name__ == "__main__":
    pytest.main(__file__)
   
