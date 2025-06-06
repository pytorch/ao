# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from functools import reduce
from typing import List, Optional

import torch

# for selecting the shards from 8 bits
maskbits = {
    1: (0x01,),
    2: (0x03,),
    3: (0x03, 0x04),
    4: (0x0F,),
    5: (0x0F, 0x10),
    6: (0x0F, 0x30),
    7: (0x0F, 0x30, 0x40),
}

unpack_mask = {
    1: (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80),
    2: (0x03, 0x0C, 0x30, 0xC0),
    4: (0x0F, 0xF0),
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
    """@AI Generated June 4, 2025
    
    Performs a left shift when shift is positive and a right shift when shift is negative.
    
    Args:
        data: The data to be shifted
        shift: The number of bits to shift (positive for left shift, negative for right shift)
        
    Returns:
        The shifted data
    """
    if shift == 0:
        return data
    elif shift < 0:
        return data >> -shift
    else:
        return data << shift


# inverse of abs_lsh for unpacking
def abs_rsh(data, shift):
    """@AI Generated June 4, 2025
    
    Performs a right shift when shift is positive and a left shift when shift is negative.
    This function is the inverse of abs_lsh.
    
    Args:
        data: The data to be shifted
        shift: The number of bits to shift (positive for right shift, negative for left shift)
        
    Returns:
        The shifted data
    """
    if shift == 0:
        return data
    elif shift < 0:
        return data << -shift
    else:
        return data >> shift


def pack_cpu(
    data: torch.Tensor, elem_size: int, dim: Optional[int] = -1
) -> List[torch.Tensor]:
    """
    Inputs:
    data: a tensor of sub byte elements in uint8
    elem_size: the size in bits of the elements to pack
    dim: the dimension to pack along
    Returns: a list of packed shards

    ==================================================================================================
    given an array such as [0x30,0x29,0x17,0x5,0x20,0x16,0x9,0x22] which are 8 uint6 elements
    first seperate into two shards: the upper 2 bits and the lower 4 bits by using a mask (0x30 and 0x0f respectively)
    2 bit shard:
    mask: 0x30
    [0x30,       0x20,       0x10,       0x00,        0x00,       0x10,       0x00,        0x20    ]
    [0b00110000, 0b00100000, 0b00010000, 0b00000000, 0b00100000, 0b00010000, 0b00000000, 0b00100000]

    Group elements into subsets that will be shifted to the same position within the 8bit container
    group1 >> 4,  group2 >> 2, group3 >> 0, group4 << 2

    [0b00000011, 0b00000010, 0b00000100, 0b00000000, 0b00100000, 0b00010000, 0b00000000, 0b10000000]
    |------ group 1 ------| |------ group 2 ------| |------ group 3 ------| |------ group 4 ------|

    Finally bitwise-or the groups together
    [0b00000011, 0b00000010,
     0b00000100, 0b00000000,
     0b00100000, 0b00010000,
     0b00000000, 0b01000000]

    [0b00100111, 0b10010010]
    ==================================================================================================
    Similarly for 4 bit shards:
    mask: 0x0f
    [0x00,       0x09,       0x07,       0x05,       0x00,       0x16,       0x9,        0x02]
    [0b00000000, 0b00001001, 0b00000111, 0b00000101, 0b00000000, 0b00000110, 0b00001001, 0b00000010]

    group1 << 0, group2 << 4
    [0b00000000, 0b00001001, 0b00000111, 0b00000101, 0b00000000, 0b01100000, 0b10010000, 0b00100000]
    |------------------ group 1 ------------------| |------------------ group 2 ------------------|

    bitwise-or:
    [0b00000000, 0b00001001, 0b00000111, 0b00000101,
     0b00000000, 0b01100000, 0b10010000, 0b00100000]

    [0b00000000, 0b01101001, 0b10010111, 0b00100101]
    ==================================================================================================
    After pack, data went from 8 elements to 6: [[0, 105, 151, 37], [39, 146]]
    In general this means pack reduces input tensor size from n * 8 to n * elem_size
    """
    torch._assert(
        data.shape[dim] % 8 == 0,
        f"pack dimension size ({data.shape[dim]}) is not divisble by scale",
    )
    torch._assert(data.dtype == torch.uint8, "data must be uint8")
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


def unpack_cpu(
    data: List[torch.Tensor], elem_size: int, dim: Optional[int] = -1
) -> torch.Tensor:
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
            output_narrow.copy_(
                torch.bitwise_or(output_narrow, abs_rsh(group, j * bit_size - rel_pos))
            )
    return output


# these are faster on the GPU


def _pack(data, elem_size, scale, dim):
    """
    Inner for loop from above pack function
    """
    packed_shape = list(data.shape)
    packed_shape[dim] = packed_shape[dim] // scale

    packed = torch.zeros(packed_shape, dtype=data.dtype, device=data.device)

    for i in range(scale):
        narrow_slice = data.narrow(
            dim, data.shape[dim] * i // scale, data.shape[dim] // scale
        )
        packed |= narrow_slice << (elem_size * i)

    return packed


def _unpack(data, element_size, scale, dim):
    """
    Inner for loop from above unpack function
    """
    unpacked_shape = list(data.shape)
    unpacked_shape[dim] *= scale

    nbits = (1 << element_size) - 1  # mask for the last element_size bits

    unpacked_data = torch.zeros(unpacked_shape, dtype=data.dtype, device=data.device)

    for i in range(scale):
        shift_amt = element_size * i
        unpacked_data.narrow(
            dim,
            unpacked_data.shape[dim] * i // scale,
            unpacked_data.shape[dim] // scale,
        ).copy_((data >> shift_amt) & nbits)

    return unpacked_data


def pack(
    data: torch.Tensor, elem_size: int, dim: Optional[int] = -1
) -> List[torch.Tensor]:
    """
    a less branching but more compute version so better for gpu
    """
    torch._assert(
        data.shape[dim] % 8 == 0,
        f"pack dimension size ({data.shape[dim]}) is not divisble by scale",
    )
    torch._assert(data.dtype == torch.uint8, "data must be uint8")
    container_size = 8
    shards = [
        (data & maskbits[elem_size][i]) >> shifts[elem_size][i]
        for i in range(len(maskbits[elem_size]))
    ]
    return tuple(
        [
            _pack(
                shards[i],
                numbits[elem_size][i],
                container_size // numbits[elem_size][i],
                dim,
            )
            for i in range(len(maskbits[elem_size]))
        ]
    )


def unpack(
    data: List[torch.Tensor], elem_size: int, dim: Optional[int] = 0
) -> torch.Tensor:
    """
    a less branching but more compute version so better for gpu
    """
    container_size = 8
    # unpack each 4,2,1 bit shard and unshift them back to the correct position
    data = [
        _unpack(
            data[i], numbits[elem_size][i], container_size // numbits[elem_size][i], dim
        )
        << shifts[elem_size][i]
        for i in range(len(data))
    ]
    return reduce(torch.bitwise_or, data)
