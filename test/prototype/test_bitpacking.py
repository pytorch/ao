import torch
from torchao.prototype.common.bitpacking import pack, unpack
import pytest
from torch.utils._triton import has_triton
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4

if not TORCH_VERSION_AFTER_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

dtypes = ((2, 'trinary', 1), (2, None, 1), (3, None, 2), (4, None, 2), (5, None, 4), (6, None, 4), (7, None, 4))
dimensions = (2, 1, 0)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_CPU(dtype, dim):
    element_bit_width, element_type,expected_pack_size = dtype
    shape = [4, 4, 4]
    if element_type == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8, device='cpu')
    else:
        test_tensor = torch.randint(0, 2**element_bit_width, shape, dtype=torch.uint8, device='cpu')
        
    packed = pack(test_tensor, element_bit_width, element_type=element_type, dim = dim, container_dtype = torch.uint8, device='cpu')
    assert(packed.shape[dim] == expected_pack_size)
    unpacked = unpack(packed, element_bit_width, element_type=element_type, dim = dim, device='cpu')
    assert(unpacked.allclose(test_tensor))

            
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_GPU(dtype, dim):
    element_bit_width, element_type,expected_pack_size = dtype
    shape = [4, 4, 4]
    if element_type == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**element_bit_width, shape, dtype=torch.uint8).cuda()
        
    packed = pack(test_tensor, element_bit_width, element_type=element_type, dim = dim, container_dtype = torch.uint8)
    assert(packed.shape[dim] == expected_pack_size)
    unpacked = unpack(packed, element_bit_width, element_type=element_type, dim = dim)
    assert(unpacked.allclose(test_tensor))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_padding(dtype, dim):
    element_bit_width, element_type,expected_pack_size = dtype
    torch._dynamo.config.specialize_int = True    
    shape =[4, 4, 4] 
    shape[dim] = 5   
    
    if element_type == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**element_bit_width, shape, dtype=torch.uint8).cuda()
        
    packed = pack(test_tensor, 
                  element_bit_width, 
                  element_type=element_type, 
                  dim = dim, 
                  container_dtype = torch.uint8,
                  pad= True)
    assert packed.shape[dim] == expected_pack_size+1, f"packed.shape[dim] {packed.shape[dim]}" # +1 for this scenario
    unpacked = unpack(packed, element_bit_width, element_type=element_type, dim = dim)
    slices = [slice(None)] * packed.ndim
    slices[dim] = slice(None, 5)
    assert unpacked[slices].allclose(test_tensor)
    
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("dim", dimensions)
def test_compile(dtype, dim):
    pack_compile = torch.compile(pack, fullgraph=True, dynamic=True)
    unpack_compile = torch.compile(unpack, fullgraph=True, dynamic=True)
    element_bit_width, element_type,expected_pack_size = dtype
    torch._dynamo.config.specialize_int = True
    shape = [4, 4, 4]
    if element_type == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**element_bit_width, shape, dtype=torch.int8).cuda()
        
    packed = pack_compile(test_tensor, element_bit_width, element_type=element_type, dim = dim, container_dtype = torch.int8)
    assert(packed.shape[dim] == expected_pack_size)
    unpacked = unpack_compile(packed, element_bit_width, element_type=element_type, dim = dim)
    assert(unpacked.allclose(test_tensor))

