import torch
from torchao.prototype.intx.bitpacking import pack, unpack
import pytest
from torch.utils._triton import has_triton

element_bit_width = (1,2,3,4,5,6,7)
dimensions = (0, -1, 1)
container_type = (torch.int8, torch.uint8, torch.int16, torch.int32)

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset() # reset cache between tests

@pytest.mark.parametrize("element_bit_width", element_bit_width)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("container_type", container_type)
def test_CPU(element_bit_width, dim, container_type):
    test_tensor = torch.randint(0, 2**element_bit_width, (32,32,32), dtype=container_type, device='cpu')
    packed = pack(test_tensor, element_bit_width, dim = dim)
    # assert(packed.shape[dim] == element_bit_width * 4 * 8 /torch.iinfo(container_type).bits)
    unpacked = unpack(packed, element_bit_width, dim = dim)
    assert(unpacked.allclose(test_tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")          
@pytest.mark.parametrize("element_bit_width", element_bit_width)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("container_type", container_type)
def test_GPU(element_bit_width, dim, container_type):
    test_tensor = torch.randint(0, 2**element_bit_width, (32,32,32), dtype=container_type).cuda()
    packed = pack(test_tensor, element_bit_width, dim = dim)
    # assert(packed.shape[dim] == element_bit_width * 4 * 8 /torch.iinfo(container_type).bits)
    unpacked = unpack(packed, element_bit_width, dim = dim)
    assert(unpacked.allclose(test_tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("element_bit_width", element_bit_width)
@pytest.mark.parametrize("dim", dimensions)
@pytest.mark.parametrize("container_type", container_type)
def test_compile(element_bit_width, dim, container_type):
    torch._dynamo.config.specialize_int = True
    pack_compile = torch.compile(pack, fullgraph=True)
    unpack_compile = torch.compile(unpack, fullgraph=True)
    test_tensor = torch.randint(0, 2**element_bit_width, (32,32,32), dtype=container_type).cuda()
    packed = pack(test_tensor, element_bit_width, dim = dim)
    # assert(packed.shape[dim] == element_bit_width * 4 * 8 /torch.iinfo(container_type).bits)
    unpacked = unpack(packed, element_bit_width, dim = dim)
    assert(unpacked.allclose(test_tensor))