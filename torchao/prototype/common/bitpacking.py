import torch
from typing import Optional, Union

def mod_shape(shape, mod, dim):
    """changes a select dimension of the input shape to mod"""
    return (*shape[:dim], mod, *shape[dim+1:])
    
def unpack(data: torch.Tensor,
           element_bit_width: int,
           element_type: Optional[str] = None, 
           dim: Optional[int] = 0,
           output_dtype: Optional[torch.dtype] = None,
           device: Optional[str] ="cuda") -> torch.Tensor:
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: - a tensor of packed elements
    element_bit_width: the size in bits of the elements to unpack
    
    optional:
    element_type: the dtype of the elements to unpack (uint,trinary,float, etc)
    dimension: the dimension to unpack along
    output_dtype: specify the dtype of the output tensor if it is not the same as the input tensor

    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    container_size = torch.iinfo(data.dtype).bits
    scale = container_size // element_bit_width
    
    unpacked = _unpack(data, element_bit_width, container_size, scale, dim, device)
    if element_type == "trinary":
        unpacked = unpacked.to(torch.int8) - 1
    elif output_dtype is not None:
        unpacked = unpacked.to(output_dtype)
        
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
         element_bit_width: int,
         element_type: Optional[str] = None,
         dim: Optional[int] = 0,
         container_dtype: Optional[torch.dtype] = None,
         pad: Optional[bool] = False,
         order: Optional[bool] = True,
         device: Optional[str] = "cuda") -> torch.Tensor:
    """
    Packs small dtype elements into a container of a larger dtype.
    **Pads rows to be divisible by the scale**
    TODO: support something like packing 8 uint 3s into 3 uint8s
    
    Inputs:
    data: a tensor of unpacked elements of a small dtype. The dtype used for the data will be used for the container.
    dim: the dimension to pack along
    element_dtype: the dtype of the elements to pack
    container_dtype: specify the dtype of the container if the data is not already inside a tensor of that dtype
    pad: if set to true, pads the dimension to be divisible by the scale
    order: if set to true, packs elements such that the lower index elements occupy the most significant bits
    
    Returns: torch.Tensor - a tensor of packed elements.
    """
    
    if element_type == "trinary":
        data =  data + 1
        
    if container_dtype is not None:
        data = data.to(container_dtype)
    
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

if __name__ == '__main__':
    pack_compile = torch.compile(pack, fullgraph=True)
    unpack_compile = torch.compile(unpack, fullgraph=True)
    torch._dynamo.config.specialize_int = True    
    element_bit_width = 2
    element_type = "trinary"
    dim = 0
    shape =[4, 4, 4] 
    shape[dim] = 5   
    
    if element_type == "trinary":
        test_tensor = torch.randint(-1, 1, shape, dtype=torch.int8).cuda()
    else:
        test_tensor = torch.randint(0, 2**element_bit_width, shape, dtype=torch.uint8).cuda()
        
    packed = pack_compile(test_tensor, element_bit_width, element_type=element_type, dim = dim, container_dtype = torch.uint8, pad= True)
    print(packed.shape)
    assert(packed.shape[dim] == 2) # +1 for this scenario
    unpacked = unpack_compile(packed, element_bit_width, element_type=element_type, dim = dim)
    slices = [slice(None)] * packed.ndim
    slices[dim] = slice(None, 5)
    print(test_tensor, "\n", packed,"\n",unpacked[slices])
    assert(unpacked[slices].allclose(test_tensor))
    
