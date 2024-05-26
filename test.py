import torch
from functools import reduce
import os

@torch.compile
def unpack(data, data_size) -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: torch.Tensor - a tensor of packed elements of a small dtype within a larger dtype.
    data_size: int - the size of the small dtype in bits.
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    shape = data.shape
    scale = data.element_size() * 8 // data_size
    unpacked_data = []
    for i in range(scale):
        shift_amt = data.element_size() * 8 - data_size * (i + 1) # how much to shift to get the ith uint
        nbits = (1 << data_size) - 1 # mask for the last dtype_size bits
        unpacked_data.append(((data >> shift_amt) & (nbits)).to(data.dtype))
    return torch.stack(unpacked_data,dim=-1).view(up_size(shape, scale)) # stack the unpacked data and reshape to the original shape

@torch.compile
def pack(data, container_size, data_size) -> torch.Tensor:
    """
    Packs small dtype elements into a larger dtype.
    
    Inputs:
    data: torch.Tensor - a tensor of unpacked elements of a small dtype.
    container_size: int - the size of the large dtype in bits.
    data_size: int - the size of the small dtype in bits.
    
    Returns: torch.Tensor - a tensor of the packed elements.
    """
    scale = container_size // data_size
    assert scale > 1, f"container_size ({container_size}) not double the capacity ofdata_size ({data_size})"
    # pad the data to be divisible by scale
    if data.shape[-1] % scale != 0:
        padding = torch.zeros((*data.shape[:-1], scale - data.shape[-1] % scale), dtype=data.dtype)
        data = torch.cat([data, padding], dim=-1)
    
    shape = data.shape
    data = data.contiguous().view(-1)
    #shift the data to the different indexes within the larger dtype and then union them together
    ret = reduce(lambda x,y: x|y,[data[i::scale] << container_size-data_size*(i+1) for i in range(scale)])
    newshape = down_size(shape, scale)
    return ret.view(newshape)

def down_size(size, amt):
    assert size[-1] % amt == 0, f"{size} last dim not divisible by {amt}"
    return (*size[:-1], size[-1] // amt)


def up_size(size, amt):
    return (*size[:-1], size[-1] * amt)


torch._dynamo.config.specialize_int = True
os.environ["TORCH_LOGS"] = "output_code"
test_tensor = torch.randint(0, 15, (1, 1, 6), dtype=torch.uint8)
packed = pack(test_tensor, 8, 4)
unpacked = unpack(packed, 4)
unpadded = unpacked[..., :test_tensor.shape[-1]]
assert(unpadded.allclose(test_tensor))

test_tensor = torch.randint(0, 7, (5,1, 4), dtype=torch.int16)
packed = pack(test_tensor,16, 3)
unpacked = unpack(packed, 3)
unpadded = unpacked[..., :test_tensor.shape[-1]]
assert(unpadded.allclose(test_tensor))

test_tensor = torch.randint(0, 15, (3,1, 9), dtype=torch.int32)
packed = pack(test_tensor,32, 16)
unpacked = unpack(packed,16)
unpadded = unpacked[..., :test_tensor.shape[-1]]
assert(unpadded.allclose(test_tensor))

test_tensor = torch.randint(0, 3, (8, 8, 7), dtype=torch.uint8)
packed = pack(test_tensor, 8, 2)
unpacked = unpack(packed,2)
unpadded = unpacked[..., :test_tensor.shape[-1]]
assert(unpadded.allclose(test_tensor))
