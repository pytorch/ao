import torch

def down_size(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by four"
    return (*size[:-1], size[-1] // 4)

def up_size(size):
    return (*size[:-1], size[-1] * 4)

@torch.compile
def unpack_uint8_to_trinary2(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 4 entries per byte
    shape = uint8_data.shape
    first_elements = ((uint8_data >> 6) & 0b11).to(torch.int8) - 1
    second_elements = ((uint8_data >> 4) & 0b11).to(torch.int8) - 1
    third_elements = ((uint8_data >> 2) & 0b11).to(torch.int8) - 1
    fourth_elements = (uint8_data & 0b11).to(torch.int8) - 1
    return torch.stack([first_elements, second_elements, third_elements, fourth_elements], dim=-1).view(up_size(shape))

@torch.compile
def pack_trinary2_to_uint8(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (uint8_data[::4] << 6 | uint8_data[1::4] << 4 | uint8_data[2::4] << 2 | uint8_data[3::4]).view(down_size(shape))
    return packed_data

def roundclip(x, a, b):
    return torch.max(a, torch.min(b, torch.round(x)))

def bitnet_clipped_quantize(weights):
    # Compute the average absolute value of the weight matrix
    gamma = torch.mean(torch.abs(weights))
    
    # Scale the weight matrix by the average absolute value
    scaled_weights = weights / (gamma + 1e-8)
    
    # Round each scaled weight to the nearest integer in {-1, 0, +1}
    quantized_weights = roundclip(scaled_weights, torch.tensor([-1]), torch.tensor([1]))
    
    return quantized_weights

def quantize_to_uint8packed_trinary2(weights):
    #Trinary packed unsigned, {-1, 0, 1} -> {0, 1, 2}
    x = (bitnet_clipped_quantize(weights) + 1.0).to(torch.uint8).to(weights.device)
    return x
