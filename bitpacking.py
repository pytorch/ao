import torch
from functools import reduce
from typing import List, Optional, Tuple
from enum import Enum
#debug
import lovely_tensors
lovely_tensors.monkey_patch()

class ZeroPointDomain(Enum):
    INT = 0
    FLOAT = 1

_DTYPE_TO_QVALUE_BOUNDS = {
    torch.uint8: (0, 255),
    torch.int8: (-128, 127),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
    torch.uint1: (0, 2**1-1),
    torch.uint2: (0, 2**2-1),
    torch.uint3: (0, 2**3-1),
    torch.uint4: (0, 2**4-1),
    torch.uint5: (0, 2**5-1),
    torch.uint6: (0, 2**6-1),
    torch.uint7: (0, 2**7-1),
}
def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """Get quant_min and quant_max args based on dtype and also
    verify that they are within the range of possible quant_min/quant_max
    for dtype
    """
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    return quant_min, quant_max

def _get_reduction_params(block_size, input_size):
    """Given block_size and input size find the parameters for reduction:

    Output:
        shape_for_reduction: the shape we use to `view` input to prepare it for reduction
        reduction_dims: the dims we'll do reduction over

    Example::
        Input:
          block_size: (3, 3, 2, 10)
          input_size: (3, 3, 10, 10)

        Output:
          shape_for_reduction: (3, 3, 5, 2, 10)
          reduction_dim: [0, 1, 3, 4]
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims

@torch.compile
def quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
):
    """
    Args:
      input (torch.Tensor): original float32, float16 or bfloat16 Tensor
      block_size: (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
           e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (float): quantization parameter for affine quantization
      zero_point (int): quantization parameter for affine quantization
      output_dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for output Tensor, if not specified, it will be derived from dtype
      quant_max (Optional[int]): maximum quantized value for output Tensor, if not specified, it will be derived from dtype
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Note:
      How can block_size represent different granularities?
      let's say we have a Tensor of size: (3, 3, 10, 10), here is the table showing how block_size represents different
      granularities:

       granularity type       |     block_size
         per_tensor           |    (3, 3, 10, 10)
         per_axis (axis=0)    |    (1, 3, 10, 10)
         per_axis (axis=1)    |    (3, 1, 10, 10)
     per_group (groupsize=2)  |    (3, 3, 10, 2)
     per_group (groupsize=2) for axis = 3 | (3, 3, 2, 10)


    Output:
      quantized tensor with requested dtype
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], f"Unsupported input dtype: {input.dtype}"
    quant_min, quant_max = _get_and_check_qmin_qmax(output_dtype, quant_min, quant_max)
    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT:
        quant = torch.clamp(
            torch.round(input * (1.0 / scale)) + zero_point, quant_min, quant_max
        ).to(output_dtype)
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT
        mid_point = (quant_max + quant_min + 1) / 2
        min_val = zero_point - scale * mid_point
        quant = (
            torch.clamp(
                torch.round((input - min_val) / scale),
                quant_min, quant_max)
        ).to(output_dtype)
    quant = quant.view(original_shape)

    return quant

@torch.compile
def dequantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    input_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    *,
    output_dtype: torch.dtype = torch.float32,
):
    """
    Args:
      input (torch.Tensor): quantized tensor, should match the dtype `dtype` argument
      block_size: (List[int]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
                               e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      scale (Tensor): quantization parameter for affine quantization
      zero_point (Tensor): quantization parameter for affine quantization
      dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor
      quant_min (Optional[int]): minimum quantized value for input Tensor
      quant_max (Optional[int]): maximum quantized value for input Tensor
      output_dtype (torch.dtype): dtype for output Tensor, default is fp32
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT

    Output:
      dequantized Tensor, with requested dtype or fp32
    """
    # TODO: validations
    # TODO: validate scale/zero_point dimensions are compatible with block_size
    assert input.dtype == input_dtype
    assert output_dtype in [torch.float32, torch.float16, torch.bfloat16], f"Unsupported output dtype: {output_dtype}"
    quant_min, quant_max = _get_and_check_qmin_qmax(input_dtype, quant_min, quant_max)

    shape_for_reduction, reduction_dims = _get_reduction_params(block_size, input.size())
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)
    if zero_point is not None:
        zero_point = zero_point.view(shape_after_reduction)

    if zero_point_domain == ZeroPointDomain.INT:
        dequant = input.to(torch.int32)
        if zero_point is not None:
            dequant -= zero_point.to(torch.int32)
        dequant = dequant.to(output_dtype)
        dequant *= scale
    else:
        assert zero_point_domain == ZeroPointDomain.FLOAT, f"Unexpected zero point domain: {zero_point_domain}"
        mid_point = (quant_max + quant_min + 1) / 2
        dequant = input - mid_point
        dequant = dequant.to(output_dtype)
        dequant *= scale
        if zero_point is not None:
            dequant += zero_point

    return dequant.view(original_shape).to(output_dtype)

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
    
    unpacked_data = torch.zeros((shape[0]*scale, *shape[1:]), dtype=data.dtype).cuda()
    nbits = (1 << data_size) - 1 # mask for the last dtype_size bits
    for i in range(scale):
        shift_amt = data.element_size() * 8 - data_size * (i + 1) # how much to shift to get the ith uint
        unpacked_data[i::scale] = ((data >> shift_amt) & (nbits))
    # print(unpacked_data)
    return unpacked_data

@torch.compile
def pack(data, container_size, data_size) -> torch.Tensor:
    """
    Packs small dtype elements into a larger dtype.
    Pads rows to be divisible by the scale.
    
    Inputs:
    data: torch.Tensor - a tensor of unpacked elements of a small dtype.
    container_size: int - the size of the large dtype in bits.
    data_size: int - the size of the small dtype in bits.
    
    Returns: torch.Tensor - a tensor of the packed elements.
    """
    scale = container_size // data_size
    assert scale > 1, f"container_size ({container_size}) is not larger than data_size ({data_size})"
    assert data.shape[0] > scale, f"not enough values to pack, data.shape[0] ({data.shape[0]}) < scale ({scale})"
    # pad the data to be divisible by scale
    if data.shape[0] % scale != 0:
        padding = torch.zeros((scale - data.shape[0] % scale, *data.shape[1:],), dtype=data.dtype).cuda()
        data = torch.cat([data, padding], dim=0).cuda()
    
    shape = data.shape
    # data = data.contiguous().view(-1)
    #shift the data to the different indexes within the larger dtype and then union them together
    # for i in range(scale):
    #     print(data[i::scale, ...])
    ret = reduce(lambda x,y: x|y,[data[i::scale, ...] << container_size-data_size*(i+1) for i in range(scale)]).cuda()
    return ret.view(shape[0] // scale, *shape[1:])



torch._dynamo.config.specialize_int = True

test_tensor = torch.randint(0, 15, (3, 4), dtype=torch.uint8).cuda()
# print("original", test_tensor)
packed = pack(test_tensor, 8, 4)
# print("packed", packed)
unpacked = unpack(packed, 4)
# print("unpacked", unpacked)
unpadded = unpacked[:test_tensor.shape[0], ...]
# print("unpadded", unpadded)
assert(unpadded.allclose(test_tensor))
print("test passed\n")
test_tensor = torch.randint(0, 7, (5, 8), dtype=torch.int16).cuda()
# print("original", test_tensor)
packed = pack(test_tensor,16, 3)
# print("packed", packed)
unpacked = unpack(packed, 3)
# print("unpacked", unpacked)   
unpadded = unpacked[:test_tensor.shape[0], ...]
# print('unpadded: ', unpadded)
assert(unpadded.allclose(test_tensor))
print("test passed\n")
test_tensor = torch.randint(0, 15, (1, 9), dtype=torch.int32).cuda()
packed = pack(test_tensor,32, 16)
print("packed", packed)
unpacked = unpack(packed,16)
# print("unpacked", unpacked) 
unpadded = unpacked[:test_tensor.shape[0], ...]
assert(unpadded.allclose(test_tensor))
print("test passed\n")
test_tensor = torch.randint(0, 3, (8, 7), dtype=torch.uint8).cuda()
packed = pack(test_tensor, 8, 2)
unpacked = unpack(packed,2)
unpadded = unpacked[:test_tensor.shape[0], ...]
assert(unpadded.allclose(test_tensor))
print("test passed\n")
test_tensor = torch.randint(0, 15, (4, 4), dtype=torch.float32).cuda()
quantize_affine(test_tensor, (4, 4), torch.tensor(0.5), torch.tensor(0), torch.uint8)
