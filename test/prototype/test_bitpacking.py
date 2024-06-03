import torch
from torchao.prototype.common.bitpacking import pack, unpack, dtype_to_bits
import pytest
from torch.utils._triton import has_triton
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4

if not TORCH_VERSION_AFTER_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

dtypes = (torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7, "trinary")
expected_pack_size = {torch.uint2: 1, torch.uint3: 2, torch.uint4: 2, torch.uint5: 4, torch.uint6: 4, torch.uint7: 4, "trinary": 1}
dimensions = (0, 1, 2)

@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_CPU(dtype, dim):
    shape = [4, 4, 4]
    if dtype == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8, device='cpu')
    else:
        test_tensor = torch.randint(0, 2**dtype_to_bits(dtype), shape, dtype=torch.uint8, device='cpu')
        
    packed = pack(test_tensor, dtype, dimension = dim, container_dtype = torch.uint8, device='cpu')
    assert(packed.shape[dim] == expected_pack_size[dtype])
    unpacked = unpack(packed, dtype, dimension = dim, device='cpu')
    assert(unpacked.allclose(test_tensor))

            
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_GPU(dtype, dim):
    shape = [4, 4, 4]
    if dtype == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**dtype_to_bits(dtype), shape, dtype=torch.uint8).cuda()
        
    packed = pack(test_tensor, dtype, dimension = dim, container_dtype = torch.uint8)
    assert(packed.shape[dim] == expected_pack_size[dtype])
    unpacked = unpack(packed, dtype, dimension = dim)
    assert(unpacked.allclose(test_tensor))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_compile(dtype, dim):
    pack_compile = torch.compile(pack, fullgraph=True)
    unpack_compile = torch.compile(unpack, fullgraph=True)
    
    shape = [4, 4, 4]
    if dtype == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**dtype_to_bits(dtype), shape, dtype=torch.uint8).cuda()
        
    packed = pack(test_tensor, dtype, dimension = dim, container_dtype = torch.uint8)
    assert(packed.shape[dim] == expected_pack_size[dtype])
    unpacked = unpack(packed, dtype, dimension = dim)
    assert(unpacked.allclose(test_tensor))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_padding(dtype, dim):
    pack_compile = torch.compile(pack, fullgraph=True)
    unpack_compile = torch.compile(unpack, fullgraph=True)
    
    shape =[4, 4, 4] 
    shape[dim] = 5   
    
    if dtype == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**dtype_to_bits(dtype), shape, dtype=torch.uint8).cuda()
        
    packed = pack(test_tensor, dtype, dimension = dim, container_dtype = torch.uint8)
    assert(packed.shape[dim] == expected_pack_size[dtype]+1) # +1 for this scenario
    unpacked = unpack(packed, dtype, dimension = dim)
    slices = [slice(None)] * packed.ndim
    slices[dim] = slice(None, 5)
    assert(unpacked[slices].allclose(test_tensor))