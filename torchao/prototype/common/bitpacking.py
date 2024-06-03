import torch
from typing import Optional, Union

def mod_shape(shape, mod, dim):
    """changes a select dimension of the input shape to mod"""
    return (*shape[:dim], mod, *shape[dim+1:])

def unpack(data: torch.Tensor,
           element_dtype: torch.dtype,
           dimension: Optional[int] = 0,
           device: Optional[str] ="cuda") -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: - a tensor of packed elements of a small dtype within a larger dtype.
    data_size: - the size of the small dtype in bits.
    
    optional:
    by_rows: bool - specifies whether to unpack... 
        by rows: tensor(n,m) -> tensor(n*scale, m) 
        or by columns: tensor(n,m) -> tensor(n,m*scale)
        
    defaults to rows because quantization is typically done by rows 
    but choose the version which matches how you quantize as this improves memory accesses/performance
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    element_size = torch.iinfo(element_dtype).bits
    container_size = torch.iinfo(data.dtype).bits
    scale = container_size // element_size
    unpacked = _unpack(data, data_size, scale, dim, device)
    if element_dtype == "trinary":
        unpacked = unpacked.to(torch.int8) - 1
    return unpacked

def _unpack(data, data_size, container_size, scale ,dim, device):
    shape = data.shape
    unpacked_data = torch.zeros(mod_shape(shape, shape[dim]*scale, dim), dtype=data.dtype).to(device)
    nbits = (1 << data_size) - 1 # mask for the last dtype_size bits
    unpacked_data = []
    for i in range(scale):
        # add the next nbits to the unpacked data
        shift_amt = container_size - data_size * (i + 1)
        unpacked_data.append(((data >> shift_amt) & (nbits)).to(data.dtype))
        
    # stack the unpacked data and reshape to the original shape
    torch.stack(unpacked_data,dim=dim).view(mod_shape(shape,scale*shape[dim], dim)) 

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
    data: - a tensor of unpacked elements of a small dtype. The dtype used for the data will be used for the container.
    dimension: - the dimension to pack along
    element_dtype: specify the dtype of the elements to pack
    
    optional:
    container_dtype: specify the dtype of the container if the data is not already inside a tensor of that dtype
        
    by_rows: specifies whether to pack values... 
        by rows: tensor(n,m) -> tensor(n//scale, m) 
        or by columns: tensor(n,m) -> tensor(n,m//scale)
    
    defaults to rows because quantization is typically done by rows
    but choose the version which matches how you quantize as this improves memory accesses/performance
    
    Returns: torch.Tensor - a tensor of packed elements.
    """
    if container_dtype is not None:
        data = data.to(container_dtype)
        
    if type(element_dtype) == str:
        if element_dtype == "trinary":
            data = data+1
        else:
            raise ValueError(f"element_dtype {element_dtype} not supported")
        
    element_size = torch.iinfo(element_dtype).bits
    container_size = torch.iinfo(data.dtype).bits
    scale = container_size // element_size
    assert data.shape[dimension] >= scale, f"not enough values to pack along dimension {dimension} ({data.shape[dimension]}) < scale ({scale})"
    return _pack_uints(data, container_size, element_dtype, scale, dimension, device)



def _pack(data, container_size, data_size, scale, dim, device) -> torch.Tensor:
    #pad dimension to be divisible by scale
    if data.shape[dimension] % scale != 0:
        padding = torch.zeros(mod_shape(data.shape, scale - data.shape[dim] % scale, dim), dtype=data.dtype).to(device)
        data = torch.cat([data, padding], dim=dim).cuda()
        
    packed = torch.zeros(mod_shape(data.shape, data.shape[dim] // scale, dim), dtype=data.dtype).to(device)
    for i in range(scale):
        torch.arange(start=i, stop=data.shape[k], step=scale)
        packed |= torch.index_select(data, dim=k, index=indices) << container_size-data_size*(i+1)
    return packed


test_tensor = torch.randint(0, 15, (4, 4), dtype=torch.uint8).cuda()
packed = pack(test_tensor, torch.uint4)
unpacked = unpack(packed, torch.uint4)
unpadded = unpacked[:test_tensor.shape[0], ...]
assert(unpadded.allclose(test_tensor))