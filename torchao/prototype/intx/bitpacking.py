import torch
import numpy as np
from typing import Optional, Union
from functools import reduce
# first value is a mask for the bits
# second value is the number of elements to pack 
# third value is the ratio shards of that size in the packed tensor
maskbits = {
    1: (0x01,),
    2: (0x03,),
    3: (0x03, 0x04),
    4: (0x0f,),
    5: (0x0f, 0x10),
    6: (0x0f, 0x30),
    7: (0x0f, 0x30, 0x40),
}

numbits = {
    1: (1,),
    2: (2,),
    3: (2, 1),
    4: (4,),
    5: (4, 1),
    6: (4, 2),
    7: (4, 2, 1),
}

shifts = {
    1: (0,),
    2: (0,),
    3: (0, 2),
    4: (0,),
    5: (0, 4),
    6: (0, 4),
    7: (0, 4, 6),
}

    
def mod_shape(shape, mod, dim):
    """changes a select dimension of the input shape to mod"""
    a = list(shape)
    a[dim] = mod
    return tuple(a)
    
def unpack(data: torch.Tensor,
           elem_size: int, 
           dim: Optional[int] = 0) -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: - a tensor of packed elements
    elem_size: the size in bits of the elements to unpack
    dim: the dimension to unpack along
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    container_size = torch.iinfo(data.dtype).bits
    shards = []
    s=0
    e=0
    m = data.shape[dim] // elem_size
    
    for i in numbits[elem_size]:
        e  = s + i
        slices = [slice(None)] * data.ndim
        slices[dim] = slice(s*m, e*m, 1)
        shards.append(slices)
        s = e 

    shards = [_unpack(data[shards[i]], numbits[elem_size][i], container_size // numbits[elem_size][i], dim) << shifts[elem_size][i] for i in range(len(shards))]
    return reduce(torch.bitwise_or, shards)

def _unpack(data, element_size, scale, dim):
    shape = data.shape
    unpacked_data = torch.zeros(mod_shape(shape, shape[dim]*scale, dim), dtype=data.dtype).to(data.device)
    nbits = (1 << element_size) - 1 # mask for the last dtype_size bits
    slices = [slice(None)] * unpacked_data.ndim
    for i in range(scale):
        shift_amt = element_size * i
        slices[dim] = slice(i, None, scale)
        unpacked_data[slices] = ((data >> shift_amt) & (nbits)).to(data.dtype)

    # stack the unpacked data and reshape to the original shape
    return unpacked_data.view(mod_shape(shape,scale*shape[dim], dim)) 
    

def pack(data: torch.Tensor,
         elem_size: int,
         dim: Optional[int] = 0) -> torch.Tensor:
    """
    Packs small dtype elements into a container of a larger dtype.
    For example, packing 4-bit elements into 8-bit containers. 
    along dimension 0:     along dimension 1:
    (0, 9,  B,  4)   -->   ( 90, 4B)                   
    (3, 8,  F,  C)   -->   (83, CF)                 
     |  |   |   |                       
     v  v   v   v                       
    (30, 89, FB, C4)
    
    Inputs:
    data: a tensor of unpacked elements of a small dtype. The dtype used for the data will be used for the container.
    dim: the dimension to pack along
    Returns: torch.Tensor - a tensor of packed elements.
    """
    container_size = torch.iinfo(data.dtype).bits
    torch._assert(data.shape[dim] % 8 == 0, f"pack dimension size ({data.shape[dim]}) is not divisble by scale")
    shards = [(data & maskbits[elem_size][i]) >> shifts[elem_size][i] for i in range(len(maskbits[elem_size]))]
    return torch.cat([_pack(shards[i], numbits[elem_size][i], container_size//numbits[elem_size][i], dim) for i in range(len(maskbits[elem_size]))],dim=dim)


def _pack(data, elem_size, scale, dim) -> torch.Tensor:
    packed = torch.zeros(mod_shape(data.shape, data.shape[dim] // scale, dim), dtype=data.dtype).to(data.device)
    slices = [slice(None)] * packed.ndim
    for i in range(scale):
        slices[dim] = slice(i, None, scale)
        packed |= data[slices] << elem_size*i
    return packed