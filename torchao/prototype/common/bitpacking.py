import torch
from functools import reduce



def unpack(data, data_size, by_rows = True, device="cuda"):
    """
    Unpacks small dtype elements from a larger dtype.
    
    Inputs:
    data: torch.Tensor - a tensor of packed elements of a small dtype within a larger dtype.
    data_size: int - the size of the small dtype in bits.
    
    optional:
    by_rows: bool - specifies whether to unpack... 
        by rows: tensor(n,m) -> tensor(n*scale, m) 
        or by columns: tensor(n,m) -> tensor(n,m*scale)
        
    defaults to rows because quantization is typically done by rows 
    but choose the version which matches how you quantize as this improves memory accesses/performance
    
    Returns: torch.Tensor - a tensor of the unpacked elements.
    """
    if by_rows:
        return _unpack_by_rows(data, data_size, device)
    else:
        return _unpack_by_cols(data, data_size)
    
def pack(data, container_size, data_size, by_rows = True, device="cuda"):
    """
    Packs small dtype elements into a larger dtype.
    Pads rows to be divisible by the scale.
    
    Inputs:
    data: torch.Tensor - a tensor of unpacked elements of a small dtype.
    container_size: int - the size of the large dtype in bits.
    data_size: int - the size of the small dtype in bits.
    
    optional:
    by_rows: bool - specifies whether to pack values... 
        by rows: tensor(n,m) -> tensor(n//scale, m) 
        or by columns: tensor(n,m) -> tensor(n,m//scale)
    
    defaults to rows because quantization is typically done by rows
    but choose the version which matches how you quantize as this improves memory accesses/performance
    
    Returns: torch.Tensor - a tensor of packed elements.
    """
    if by_rows:
        return _pack_by_rows(data, container_size, data_size, device)
    else:
        return _pack_by_cols(data, container_size, data_size, device)   
    
def _unpack_by_rows(data, data_size, device) -> torch.Tensor:
    shape = data.shape
    scale = data.element_size() * 8 // data_size
    
    unpacked_data = torch.zeros((shape[0]*scale, *shape[1:]), dtype=data.dtype).to(device)
    nbits = (1 << data_size) - 1 # mask for the last dtype_size bits
    for i in range(scale):
        shift_amt = data.element_size() * 8 - data_size * (i + 1) # how much to shift to get the ith uint
        unpacked_data[i::scale] = ((data >> shift_amt) & (nbits))
    return unpacked_data

def _unpack_by_cols(data, data_size) -> torch.Tensor:
    shape = data.shape
    scale = data.element_size() * 8 // data_size
    unpacked_data = []
    nbits = (1 << data_size) - 1 # mask for the last dtype_size bits
    for i in range(scale):
        shift_amt = data.element_size() * 8 - data_size * (i + 1) # how much to shift to get the ith uint
        unpacked_data.append(((data >> shift_amt) & (nbits)).to(data.dtype))
    return torch.stack(unpacked_data,dim=-1).view(*shape[:-1],shape[-1]*scale) # stack the unpacked data and reshape to the original shape

def _pack_by_rows(data, container_size, data_size, device) -> torch.Tensor:
    
    scale = container_size // data_size
    assert scale > 1, f"container_size ({container_size}) is not larger than data_size ({data_size})"
    assert data.shape[0] >= scale, f"not enough values to pack, data.shape[0] ({data.shape[0]}) < scale ({scale})"
    # pad the data to be divisible by scale
    if data.shape[0] % scale != 0:
        padding = torch.zeros((scale - data.shape[0] % scale, *data.shape[1:],), dtype=data.dtype).to(device)
        data = torch.cat([data, padding], dim=0).cuda()
    
    shape = data.shape
    ret = reduce(lambda x,y: x|y,[data[i::scale, ...] << container_size-data_size*(i+1) for i in range(scale)])
    return ret.view(shape[0] // scale, *shape[1:]).to(device)

def _pack_by_cols(data, container_size, data_size, device) -> torch.Tensor:
    scale = container_size // data_size
    assert scale > 1, f"container_size ({container_size}) not double the capacity ofdata_size ({data_size})"
    # pad the data to be divisible by scale
    if data.shape[-1] % scale != 0:
        padding = torch.zeros((*data.shape[:-1], scale - data.shape[-1] % scale), dtype=data.dtype).to(device)
        data = torch.cat([data, padding], dim=-1).cuda()
    
    shape = data.shape
    data = data.contiguous().view(-1)
    #shift the data to the different indexes within the larger dtype and then union them together
    ret = reduce(lambda x,y: x|y,[data[i::scale] << container_size-data_size*(i+1) for i in range(scale)])
    return ret.view(*shape[:-1],shape[-1] // scale).to(device)