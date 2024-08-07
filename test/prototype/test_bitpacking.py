import torch
from torchao.prototype.uintx import pack, unpack, pack_cpu, unpack_cpu
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
    packed = pack_cpu(test_tensor, element_bit_width, dim = dim)
    unpacked = unpack_cpu(packed, element_bit_width, dim = dim)
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

# these test cases are for the example pack walk through in the bitpacking.py file
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pack_example():
    test_tensor = torch.tensor([0x30,0x29,0x17,0x5,0x20,0x16,0x9,0x22], dtype=torch.uint8).cuda()
    shard_4,shard_2  = pack(test_tensor, 6)
    print(shard_4, shard_2)
    assert torch.tensor([0, 105, 151, 37], dtype=torch.uint8).cuda().allclose(shard_4)
    assert torch.tensor([39, 146], dtype=torch.uint8).cuda().allclose(shard_2)
    unpacked = unpack([shard_4, shard_2], 6)
    assert unpacked.allclose(test_tensor)

def test_pack_example_CPU():
    test_tensor = torch.tensor([0x30,0x29,0x17,0x5,0x20,0x16,0x9,0x22], dtype=torch.uint8)
    shard_4,shard_2  = pack(test_tensor, 6)
    print(shard_4, shard_2)
    assert torch.tensor([0, 105, 151, 37], dtype=torch.uint8).allclose(shard_4)
    assert torch.tensor([39, 146], dtype=torch.uint8).allclose(shard_2)
    unpacked = unpack([shard_4, shard_2], 6)
    assert unpacked.allclose(test_tensor)
    
    