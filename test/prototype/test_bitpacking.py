import torch
from torchao.prototype.common.bitpacking import pack, unpack, dtype_to_bits
import pytest
from torch.utils._triton import has_triton
from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4

if not TORCH_VERSION_AFTER_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

def test_trinary_to_uint8_CPU():
    test_tensor = torch.randint(-1, 1, (4, 4, 4), dtype=torch.int32)
    for i in range(len(test_tensor.shape)):
        packed = pack(test_tensor, "trinary", dimension = i, container_dtype = torch.uint8, device='cpu')
        unpacked = unpack(packed, "trinary", dimension = i, device='cpu')
        assert(unpacked.to(torch.int32).allclose(test_tensor))

def test_to_uint8_CPU():
    for dtype in {torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7}:
        test_tensor = torch.randint(0, 2**dtype_to_bits(dtype), (4, 4, 4), dtype=torch.uint8)
        for i in range(len(test_tensor.shape)):
            packed = pack(test_tensor, dtype, dimension = i, container_dtype = torch.uint8, device='cpu')
            unpacked = unpack(packed, dtype, dimension = i, device='cpu')
            assert unpacked.allclose(test_tensor), f"Failed for {dtype} on dim {i}"
            
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trinary_to_uint8():
    test_tensor = torch.randint(-1, 1, (4, 4, 4), dtype=torch.int32).cuda()
    for i in range(len(test_tensor.shape)):
        packed = pack(test_tensor, "trinary", dimension = i, container_dtype = torch.uint8)
        unpacked = unpack(packed, "trinary", dimension = i)
        assert(unpacked.to(torch.int32).allclose(test_tensor))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_uint8():
    for dtype in {torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7}:
        test_tensor = torch.randint(0, 2**dtype_to_bits(dtype), (4, 4, 4), dtype=torch.uint8).cuda()
        for i in range(len(test_tensor.shape)):
            packed = pack(test_tensor, dtype, dimension = i, container_dtype = torch.uint8)
            unpacked = unpack(packed, dtype, dimension = i)
            assert unpacked.allclose(test_tensor), f"Failed for {dtype} on dim {i}"
            
test_trinary_to_uint8_CPU()
test_to_uint8_CPU()
test_trinary_to_uint8()
test_to_uint8()