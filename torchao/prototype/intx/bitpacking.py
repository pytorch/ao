import torch
import numpy as np
from typing import Optional, Union, List
from functools import reduce

# for selecting the shards from 8 bits
maskbits = {
    1: (0x01,),
    2: (0x03,),
    3: (0x03, 0x04),
    4: (0x0f,),
    5: (0x0f, 0x10),
    6: (0x0f, 0x30),
    7: (0x0f, 0x30, 0x40),
}

unpack_mask = {
    1: (0x01,0x02,0x04,0x08, 0x10,0x20,0x40,0x80),
    2: (0x03,0x0c,0x30,0xc0),
    4: (0x0f,0xf0),
}

# size of each shard
numbits = {
    1: (1,),
    2: (2,),
    3: (2, 1),
    4: (4,),
    5: (4, 1),
    6: (4, 2),
    7: (4, 2, 1),
}

# shift amount for each shard
shifts = {
    1: (0,),
    2: (0,),
    3: (0, 2),
    4: (0,),
    5: (0, 4),
    6: (0, 4),
    7: (0, 4, 6),
}

# for shifting groups left but right if shift is negative
def abs_lsh(data, shift):
    if shift == 0:
        return data
    elif shift < 0:
        return data >> -shift
    else:
        return data << shift


# inverse of abs_lsh for unpacking
def abs_rsh(data, shift):
    if shift == 0:
        return data
    elif shift < 0:
        return data << -shift
    else:
        return data >> shift


def pack_cpu(data: torch.Tensor,
         elem_size: int,
         dim: Optional[int] = -1) -> List[torch.Tensor]:
    """
    Inputs:
    data: a tensor of sub byte elements in uint8 
    dim: the dimension to pack along
    Returns: a list of packed shards
    
    given an array such as [0x30,0x29,0x17,0x5,0x20,0x16,0x9,0x2] which are 8 uint6 elements
    first seperate into two shards: the upper 2 bits and the lower 4 bits by using a mask (0x30 and 0x0f respectively)
    2 bit shards: [0b00110000, 0b00100000, 0b00010000, 0b00000000, 0b00100000, 0b00010000, 0b00000000, 0b00000000]
                   |------ group 1 ------| |------ group 2 ------| |------ group 3 ------| |------ group 4 ------|
    4 bit shards: [0b00000000, 0b00001001, 0b00000111, 0b00000101, 0b00000000, 0b00000110, 0b00001001, 0b00000010]
                   |------------------ group 1 ------------------| |------------------ group 2 ------------------|
    Then pack each of these shards by shifting each group of 4 uint2s or 2 uint4s to a different position within the 8bit container
    2bit shard group1 >> 4,  group2 >> 2, group3 >> 0, group4 << 2
    [0b00000011, 0b00000001, 0b00000100, 0b00000000, 0b00100000, 0b00010000, 0b00000000, 0b00000000]
    |------ group 1 ------| |------ group 2 ------| |------ group 3 ------| |------ group 4 ------|
    4 bit shard group1 << 0, group2 << 4
    [0b00000000, 0b00001001, 0b00000111, 0b00000101, 0b00000000, 0b01100000, 0b10010000, 0b00100000]
    |------------------ group 1 ------------------| |------------------ group 2 ------------------|
    finally bitwise or the groups together
    2 bit shard: [0b00100111, 00010001]
    4 bit shard: [0b00000000, 0b01101001, 0b10010111, 0b00100101]
    so we went from 8 elements to 6
    in general this means we go from 8 * n to elem_size * n
    """
    torch._assert(data.shape[dim] % 8 == 0, f"pack dimension size ({data.shape[dim]}) is not divisble by scale")
    if data.dtype != torch.uint8:
        data = data.to(torch.uint8)
    output_shape = list(data.shape)
    
    output = []
    for i in range(len(numbits[elem_size])):
        output_shape[dim] = data.shape[dim] * numbits[elem_size][i] // 8
        shard = torch.zeros(output_shape, dtype=torch.uint8, device=data.device)
        bit_size = numbits[elem_size][i]
        rel_pos = shifts[elem_size][i]
        bits = data & maskbits[elem_size][i]
        scale = 8 // bit_size
        slice_len = bits.shape[dim] // scale
        for j in range(scale):
            bit_slice = bits.narrow(dim, slice_len * j, slice_len)
            shard = torch.bitwise_or(shard, abs_lsh(bit_slice, j * bit_size - rel_pos))
        output.append(shard)
    return output


def unpack_cpu(data: List[torch.Tensor],
           elem_size: int, 
           dim: Optional[int] = -1) -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: - a list of packed shards
    elem_size: the size in bits of the elements to unpack
    dim: the dimension to unpack along
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    # define the output tensor
    output_shape = list(data[0].shape)
    output_shape[dim] = data[0].shape[dim] * 8 // numbits[elem_size][0]
    output = torch.zeros(output_shape, dtype=torch.uint8, device=data[0].device)
    
    for i in range(len(numbits[elem_size])):
        # define variables for the current shard
        bit_size = numbits[elem_size][i]
        rel_pos = shifts[elem_size][i]
        scale = 8 // bit_size
        group_size = bit_size * output_shape[dim] // 8
        # mask and shift every group of bits to the correct position
        for j in range(scale):
            output_narrow = output.narrow(dim, j * group_size, group_size)
            group = data[i] & unpack_mask[bit_size][j]
            shift_amt = j * bit_size - rel_pos
            output_narrow.copy_(torch.bitwise_or(output_narrow, abs_rsh(group, j * bit_size - rel_pos)))
    return output 

# these are faster on the GPU

def _pack(data, elem_size, scale, dim):
    '''
    Inner for loop from above pack function
    '''
    packed_shape = list(data.shape)
    packed_shape[dim] = packed_shape[dim] // scale
    
    packed = torch.zeros(packed_shape, dtype=data.dtype, device=data.device)
    
    for i in range(scale):
        narrow_slice = data.narrow(dim, data.shape[dim]*i//scale, data.shape[dim] // scale)
        packed |= narrow_slice << (elem_size * i)
    
    return packed

def _unpack(data, element_size, scale, dim):
    '''
    Inner for loop from above unpack function
    '''
    unpacked_shape = list(data.shape)
    unpacked_shape[dim] *= scale
    
    nbits = (1 << element_size) - 1  # mask for the last element_size bits
    
    unpacked_data = torch.zeros(unpacked_shape, dtype=data.dtype, device=data.device)
    
    for i in range(scale):
        shift_amt = element_size * i
        chunk = unpacked_data.narrow(dim, unpacked_data.shape[dim]*i//scale, unpacked_data.shape[dim] // scale).copy_((data >> shift_amt) & nbits)
    
    return unpacked_data


def pack(data: torch.Tensor,
         elem_size: int,
         dim: Optional[int] = -1) -> List[torch.Tensor]:
    ''''''
    container_size = torch.iinfo(data.dtype).bits
    torch._assert(data.shape[dim] % 8 == 0, f"pack dimension size ({data.shape[dim]}) is not divisble by scale")
    shards = [(data & maskbits[elem_size][i]) >> shifts[elem_size][i] for i in range(len(maskbits[elem_size]))]
    return [_pack(shards[i], numbits[elem_size][i], container_size//numbits[elem_size][i], dim) for i in range(len(maskbits[elem_size]))]

def unpack(shards: List[torch.Tensor],
                  elem_size: int,
                  dim: Optional[int] = 0) -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    """
    container_size = torch.iinfo(shards[0].dtype).bits
    # unpack each 4,2,1 bit shard and unshift them back to the correct position
    shards = [_unpack(shards[i], numbits[elem_size][i], container_size // numbits[elem_size][i], dim) << shifts[elem_size][i] for i in range(len(shards))]
    return reduce(torch.bitwise_or, shards)


if __name__ == "__main__":
    fake_tensor = torch.arange(64, dtype=torch.uint8).view(8,8)
    # print(fake_tensor)
    packed = pack(fake_tensor, 6)
    # print(packed)
    unpacked = unpack(packed, 6)
    # print(unpacked)
    assert torch.all(fake_tensor == unpacked)