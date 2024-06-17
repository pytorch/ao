import pytest
import torch
from torchao.prototype.dtypes import BitnetTensor
from torchao.prototype.dtypes.uint2 import unpack_uint2

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

if __name__ == "__main__":
    pytest.main()
   
