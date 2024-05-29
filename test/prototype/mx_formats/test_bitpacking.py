import torch
from torchao.prototype.common.bitpacking import pack, unpack
import pytest


def test_uint4_to_uint8():
    test_tensor = torch.randint(0, 15, (4, 4), dtype=torch.uint8).cuda()
    packed = pack(test_tensor, 8, 4)
    unpacked = unpack(packed, 4)
    unpadded = unpacked[:test_tensor.shape[0], ...]
    assert(unpadded.allclose(test_tensor))

def test_uint4_to_uint8_compile():
    torch._dynamo.config.specialize_int = True
    pack = torch.compile(pack)
    unpack = torch.compile(unpack)
    test_tensor = torch.randint(0, 15, (3, 4), dtype=torch.uint8).cuda()
    packed = pack(test_tensor, 8, 4)
    unpacked = unpack(packed, 4)
    unpadded = unpacked[:test_tensor.shape[0], ...]
    assert(unpadded.allclose(test_tensor))

def test_uint3_to_int16():
    test_tensor = torch.randint(0, 7, (5, 8), dtype=torch.int16).cuda()
    packed = pack(test_tensor,16, 3)
    unpacked = unpack(packed, 3)
    unpadded = unpacked[:test_tensor.shape[0], ...]
    assert(unpadded.allclose(test_tensor))


def test_uint2_to_uint8_col_wise_compile():
    torch._dynamo.config.specialize_int = True
    pack = torch.compile(pack)
    unpack = torch.compile(unpack)
    test_tensor = torch.randint(0, 3, (8, 8), dtype=torch.uint8).cuda()
    packed = pack(test_tensor, 8, 2, False)
    unpacked = unpack(packed,2, False)
    unpadded = unpacked[:test_tensor.shape[0], ...]
    assert(unpadded.allclose(test_tensor))

def test_uint3_to_int16_col_wise():
    test_tensor = torch.randint(0, 7, (8, 5), dtype=torch.int16).cuda()
    packed = pack(test_tensor,16, 3, False)
    unpacked = unpack(packed, 3, False)
    unpadded = unpacked[:test_tensor.shape[0], ...]
    assert(unpadded.allclose(test_tensor))