import torch
from torchao.prototype.intx.bitpacking import pack, unpack
import pytest
from torch.utils._triton import has_triton

element_bit_width = (1,2,3,4,5,6,7)
dimensions = (0, -1, 1)

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset() # reset cache between tests

@pytest.mark.parametrize("element_bit_width", element_bit_width)
@pytest.mark.parametrize("dim", dimensions)
def test_CPU(element_bit_width, dim):
    test_tensor = torch.randint(0, 2**element_bit_width, (32,32,32), dtype=torch.uint8, device='cpu')
    packed = pack(test_tensor, element_bit_width, dim = dim)
    unpacked = unpack(packed, element_bit_width, dim = dim)
    assert(unpacked.allclose(test_tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")          
@pytest.mark.parametrize("element_bit_width", element_bit_width)
@pytest.mark.parametrize("dim", dimensions)
def test_GPU(element_bit_width, dim):
    test_tensor = torch.randint(0, 2**element_bit_width, (32,32,32), dtype=torch.uint8).cuda()
    packed = pack(test_tensor, element_bit_width, dim = dim)
    unpacked = unpack(packed, element_bit_width, dim = dim)
    assert(unpacked.allclose(test_tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("element_bit_width", element_bit_width)
@pytest.mark.parametrize("dim", dimensions)
def test_compile(element_bit_width, dim):
    torch._dynamo.config.specialize_int = True
    pack_compile = torch.compile(pack, fullgraph=True)
    unpack_compile = torch.compile(unpack, fullgraph=True)
    test_tensor = torch.randint(0, 2**element_bit_width, (32,32,32), dtype=torch.uint8).cuda()
    packed = pack(test_tensor, element_bit_width, dim = dim)
    unpacked = unpack(packed, element_bit_width, dim = dim)
    assert(unpacked.allclose(test_tensor))