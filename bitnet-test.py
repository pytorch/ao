import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch.library import impl, Library
import lovely_tensors as lt
lt.monkey_patch()

def down_size(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by four"
    return (*size[:-1], size[-1] // 4)

def up_size(size):
    return (*size[:-1], size[-1] * 4)

def unpack_uint2(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 4 entries per byte
    shape = uint8_data.shape
    unpacked_data = torch.empty((*shape, 4), dtype=torch.uint8)

    unpacked_data[..., 0] = (uint8_data >> 6) & 0b11
    unpacked_data[..., 1] = (uint8_data >> 4) & 0b11
    unpacked_data[..., 2] = (uint8_data >> 2) & 0b11
    unpacked_data[..., 3] = uint8_data & 0b11
    return unpacked_data.view(up_size(shape))

def pack_uint2(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (uint8_data[::4] << 6 | uint8_data[1::4] << 4 | uint8_data[2::4] << 2 | uint8_data[3::4]).view(down_size(shape))
    return packed_data

def roundclip(x, a, b):
    return torch.max(torch.tensor(a), torch.min(torch.tensor(b), torch.round(x)))

def quantize_per_tensor_uint2_trinary(weights):
    # Compute the average absolute value of the weight tensor
    gamma = torch.mean(torch.abs(weights))
    
    # Scale the weight tensor by the average absolute value
    scaled_weights = weights / (gamma + 1e-8)
    
    # Round each scaled weight to the nearest integer in {-1, 0, +1}
    quantized_weights = roundclip(scaled_weights, -1, 1)

    #Shift the distribution over by 1 so we can pack into a uint and not deal with signs
    return quantized_weights.to(torch.int8)

test_tensor = torch.randint(0, 3, (1024, 16, 8), dtype=torch.uint8)
print(test_tensor)
packed = pack_uint2(test_tensor)
unpacked = unpack_uint2(packed)
print(unpacked.allclose(test_tensor))
assert(unpacked.allclose(test_tensor))

test_layer = torch.rand(1024, 16, 8) * 500.0 - 250.0

#Quantize our fake layer with bitnet method.
original_fake_layer = quantize_per_tensor_uint2_trinary(test_layer)
print(original_fake_layer)

#Shift distribution from -1, 1 -> 0, 2 to we can use unsigned storage.
shifted_fake_layer = (original_fake_layer + 1.0).to(torch.uint8)
print("original: ")
print(shifted_fake_layer)



def unpack_uint8_to_trinary(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 4 entries per byte
    shape = uint8_data.shape
    unpacked_data = torch.empty((*shape, 4), dtype=torch.int8)

    unpacked_data[..., 0] = ((uint8_data >> 6) & 0b11).to(torch.int8) - 1.0
    unpacked_data[..., 1] = ((uint8_data >> 4) & 0b11).to(torch.int8) - 1.0
    unpacked_data[..., 2] = ((uint8_data >> 2) & 0b11).to(torch.int8) - 1.0
    unpacked_data[..., 3] = (uint8_data & 0b11).to(torch.int8) - 1.0
    return unpacked_data.view(up_size(shape))

def pack_uint2(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (uint8_data[::4] << 6 | uint8_data[1::4] << 4 | uint8_data[2::4] << 2 | uint8_data[3::4]).view(down_size(shape))
    return packed_data


packed = pack_uint2(shifted_fake_layer)
print("after packing: ")
print(packed)
unpacked = unpack_uint8_to_trinary(packed)
print("after unpacking: ")
print(unpacked)
print(unpacked.dtype)
print(unpacked.allclose(original_fake_layer))
assert(unpacked.allclose(original_fake_layer))

unpack_empty = torch.compile(unpack_uint8_to_trinary, mode="reduce-overhead")