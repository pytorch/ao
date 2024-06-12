import torch
from typing import Optional, Union

def mod_shape(shape, mod, dim):
    """changes a select dimension of the input shape to mod"""
    a = list(shape)
    a[dim] = mod
    return tuple(a)
    
def unpack(data: torch.Tensor,
           element_bit_width: int,
           element_type: Optional[str] = None, 
           dim: Optional[int] = 0,
           order: Optional[bool] = True,
           output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: - a tensor of packed elements
    element_bit_width: the size in bits of the elements to unpack
    element_type: the dtype of the elements to unpack (uint,trinary,float, etc)
    dim: the dimension to unpack along
    output_dtype: specify the dtype of the output tensor if it is not the same as the input tensor
    order: make sure it matches the value set in the pack function
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    container_size = torch.iinfo(data.dtype).bits
    scale = container_size // element_bit_width
    device = data.device
        
    unpacked = _unpack(data, element_bit_width, container_size, scale, order, dim, device)
    
    if element_type == "trinary":
        unpacked = unpacked.to(torch.int8) - 1
    elif output_dtype is not None:
        unpacked = unpacked.to(output_dtype)
        
    return unpacked

def _unpack(data, element_size, container_size, scale, order, dim, device):
    shape = data.shape
    unpacked_data = torch.zeros(mod_shape(shape, shape[dim]*scale, dim), dtype=data.dtype).to(device)
    nbits = (1 << element_size) - 1 # mask for the last dtype_size bits
    for i in range(scale):
        if order:
            shift_amt = container_size - element_size * (i + 1)
        else:
            shift_amt = element_size * i
        slices = [slice(None)] * unpacked_data.ndim
        slices[dim] = slice(i, None, scale)
        unpacked_data[slices] = ((data >> shift_amt) & (nbits)).to(data.dtype)

    # stack the unpacked data and reshape to the original shape
    return unpacked_data.view(mod_shape(shape,scale*shape[dim], dim)) 
    

def pack(data: torch.Tensor,
         element_bit_width: int,
         element_type: Optional[str] = None,
         dim: Optional[int] = 0,
         container_dtype: Optional[torch.dtype] = None,
         pad: Optional[bool] = False,
         order: Optional[bool] = True) -> torch.Tensor:
    """
    Packs small dtype elements into a container of a larger dtype.
    
    Inputs:
    data: a tensor of unpacked elements of a small dtype. The dtype used for the data will be used for the container.
    dim: the dimension to pack along
    element_dtype: the dtype of the elements to pack
    container_dtype: specify the dtype of the container if the data is not already inside a tensor of that dtype
    pad: if set to true, pads the dimension to be divisible by the scale
    order: if set to true, packs elements such that the lower index elements occupy the most significant bits
    
    Returns: torch.Tensor - a tensor of packed elements.
    
    
    For example, packing 4-bit elements into 8-bit containers. 
    along dimension 0:     along dimension 1:
    (0, 9,  B,  4)   -->   ( 9, B4)                   
    (3, 8,  F,  C)   -->   (38, FC)                 
     |  |   |   |                       
     v  v   v   v                       
    (3, 98, BF, 4C)
    
    if order was set to false:
    (30, 89, FB, C4)
    """
    
    if element_type == "trinary":
        data =  data + 1
        
    if container_dtype is not None:
        data = data.to(container_dtype)
    
    device = data.device
    
    container_size = torch.iinfo(data.dtype).bits
    scale = container_size // element_bit_width
    
    if pad and data.shape[dim] % scale != 0:
        padding = torch.zeros(mod_shape(data.shape, scale - data.shape[dim] % scale, dim), dtype=data.dtype).to(device)
        data = torch.cat([data, padding], dim=dim).to(device)
        
    
    torch._assert(data.shape[dim] >= scale, f"not enough values to pack along dimension {dim}")
    torch._assert(data.shape[dim] % scale == 0, "size of pack dimension not divisble by scale")
    return _pack(data, container_size, element_bit_width, scale, dim, order, device)



def _pack(data, container_size, element_bit_width, scale, dim, order, device) -> torch.Tensor:
    packed = torch.zeros(mod_shape(data.shape, data.shape[dim] // scale, dim), dtype=data.dtype).to(device)
    slices = [slice(None)] * packed.ndim
    for i in range(scale):
        slices[dim] = slice(i, None, scale)
        if order:
            packed |= data[slices] << container_size-element_bit_width*(i+1)
        else:
            packed |= data[slices] << element_bit_width*i
    return packed
    