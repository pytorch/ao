import torch
from typing import Optional, Union

def mod_shape(shape, mod, dim):
    """changes a select dimension of the input shape to mod"""
    return (*shape[:dim], mod, *shape[dim+1:])


def dtype_to_bits(dtype):
    '''returns the number of bits in a dtype'''
    if dtype in {torch.uint2, 'trinary'}:
        return 2
    elif dtype == torch.uint3:
        return 3
    elif dtype == torch.uint4:
        return 4
    elif dtype == torch.uint5:
        return 5
    elif dtype == torch.uint6:
        return 6
    elif dtype == torch.uint7:
        return 7
    elif dtype in {torch.uint8, torch.int8}:
        return 8
    elif dtype in {torch.uint16, torch.int16, torch.float16}:
        return 16
    elif dtype in {torch.uint32, torch.int32, torch.float32}:
        return 32
    elif dtype == {torch.uint64, torch.int64, torch.float64}:
        return 64
    else:
        raise ValueError(f"dtype {dtype} not supported (yet)")
    
def unpack(data: torch.Tensor,
           element_dtype: Union[torch.dtype, str], # accepting strings for trinary until thats added to torch
           dimension: Optional[int] = 0,
           device: Optional[str] ="cuda") -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: - a tensor of packed elements
    element_dtype: - the dtype of the elements to unpack
    
    optional:
    dimension: - the dimension to unpack along
    
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    container_size = dtype_to_bits(data.dtype)
    element_size = dtype_to_bits(element_dtype)
    scale = container_size // element_size
    
    unpacked = _unpack(data, element_size, container_size, scale, dimension, device)
    
    if element_dtype == "trinary":
        unpacked = unpacked.to(torch.int8) - 1
    return unpacked

def _unpack(data, element_size, container_size, scale ,dim, device):
    shape = data.shape
    unpacked_data = torch.zeros(mod_shape(shape, shape[dim]*scale, dim), dtype=data.dtype).to(device)
    nbits = (1 << element_size) - 1 # mask for the last dtype_size bits
    for i in range(scale):
        shift_amt = container_size - element_size * (i + 1)
        slices = [slice(None)] * unpacked_data.ndim
        slices[dim] = slice(i, None, scale)
        unpacked_data[slices] = ((data >> shift_amt) & (nbits)).to(data.dtype)

    # stack the unpacked data and reshape to the original shape
    return unpacked_data.view(mod_shape(shape,scale*shape[dim], dim)) 
    

def pack(data: torch.Tensor,
         element_dtype: Union[torch.dtype, str], # accepting strings for trinary until thats added to torch
         dimension: Optional[int] = 0,
         container_dtype: Optional[torch.dtype] = None,
         device: Optional[str] = "cuda") -> torch.Tensor:
    """
    Packs small dtype elements into a container of a larger dtype.
    **Pads rows to be divisible by the scale**
    TODO: support something like packing 8 uint 3s into 3 uint8s
    
    Inputs:
    data: a tensor of unpacked elements of a small dtype. The dtype used for the data will be used for the container.
    dimension: the dimension to pack along
    element_dtype: the dtype of the elements to pack
    
    optional:
    container_dtype: specify the dtype of the container if the data is not already inside a tensor of that dtype
    
    
    defaults to rows because quantization is typically done by rows
    but choose the version which matches how you quantize as this improves memory accesses/performance
    
    Returns: torch.Tensor - a tensor of packed elements.
    """
    if element_dtype == "trinary":
        data =  data + 1
        
    if container_dtype is not None:
        data = data.to(container_dtype)
    
    container_size = dtype_to_bits(data.dtype)
    element_size = dtype_to_bits(element_dtype)
    scale = container_size // element_size
        
    assert data.shape[dimension] >= scale, f"not enough values to pack along dimension {dimension} ({data.shape[dimension]}) < scale ({scale})"
    return _pack(data, container_size, element_size, scale, dimension, device)



def _pack(data, container_size, element_size, scale, dim, device) -> torch.Tensor:
    #pad dimension to be divisible by scale
    if data.shape[dim] % scale != 0:
        padding = torch.zeros(mod_shape(data.shape, scale - data.shape[dim] % scale, dim), dtype=data.dtype).to(device)
        data = torch.cat([data, padding], dim=dim).to(device)
        
    packed = torch.zeros(mod_shape(data.shape, data.shape[dim] // scale, dim), dtype=data.dtype).to(device)
    for i in range(scale):
        slices = [slice(None)] * packed.ndim
        slices[dim] = slice(i, None, scale)
        packed |= data[slices] << container_size-element_size*(i+1)
    return packed