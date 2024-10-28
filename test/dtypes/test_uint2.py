import pytest
import torch
import torch.nn as nn
from torchao.prototype.dtypes import UInt2Tensor
from torchao.prototype.dtypes.uint2 import unpack_uint2
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

@pytest.fixture
def uint2_tensor():
    input_tensor = torch.randint(0, 15, (4,4), dtype = torch.uint8)
    return UInt2Tensor(input_tensor)

def test_copy(uint2_tensor):
    copied_tensor = uint2_tensor.clone()
    assert torch.equal(uint2_tensor.elem, copied_tensor.elem)

def test_transpose(uint2_tensor):
    transposed_tensor = uint2_tensor.t()
    expected_tensor = unpack_uint2(uint2_tensor.elem).t()
    assert torch.equal(unpack_uint2(transposed_tensor.elem), expected_tensor)

@pytest.mark.parametrize("dtype", [torch.float, torch.float16, torch.bfloat16, torch.int16, torch.int32, torch.int64])
def test_conversion(uint2_tensor, dtype):
    converted_tensor = uint2_tensor.to(dtype)
    expected_tensor = unpack_uint2(uint2_tensor.elem).to(dtype)
    assert torch.allclose(converted_tensor, expected_tensor, atol=1e-5)

if __name__ == '__main__':
    pytest.main(__file__)
    
