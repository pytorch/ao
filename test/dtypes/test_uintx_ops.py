import pytest
import torch
from torchao.dtypes.uintx.uintx import UintxTensor, to_uintx, _DTYPE_TO_BIT_WIDTH

# Define the dtypes to test
if torch.__version__ >= "2.3":
    dtypes = (torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7)
else:
    dtypes = ()

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

def get_bit_width_from_tensor(tensor):
    max_value = tensor.max().item()
    return max(2, (max_value + 1).bit_length())

def quantize_for_dtype(value, dtype):
    if dtype == torch.uint8:
        return value  # No quantization needed for uint8
    bit_width = _DTYPE_TO_BIT_WIDTH[dtype]
    return min(value, 2**bit_width - 1)

@pytest.fixture(params=dtypes)
def dtype(request):
    return request.param

@pytest.fixture(params=devices)
def device(request):
    return request.param

@pytest.fixture(params=list(_DTYPE_TO_BIT_WIDTH.keys()))
def uintx_tensor_and_dtype(request):
    dtype = request.param
    original_data = torch.tensor([10, 25, 40, 55, 5, 20, 35, 50], dtype=torch.uint8)
    quantized_data = torch.tensor([quantize_for_dtype(v.item(), dtype) for v in original_data], dtype=torch.uint8)
    uintx_tensor = to_uintx(quantized_data, dtype)
    return uintx_tensor, dtype


def test_basic_slicing(uintx_tensor_and_dtype):
    uintx_tensor, dtype = uintx_tensor_and_dtype
    sliced_uintx = uintx_tensor[2:6]
    sliced_data = sliced_uintx.get_plain()
    bit_width = get_bit_width_from_tensor(sliced_data)
    assert torch.all(sliced_data == sliced_data), f"Sanity check failed for {bit_width}-bit tensor"

def test_step_slicing(uintx_tensor_and_dtype):
    uintx_tensor, dtype = uintx_tensor_and_dtype
    step_sliced_uintx = uintx_tensor[1::2]
    step_sliced_data = step_sliced_uintx.get_plain()
    
    original_data = uintx_tensor.get_plain()
    expected_step_slice = original_data[1::2]
    
    expected_step_slice = expected_step_slice.to(step_sliced_data.dtype)
    
    assert torch.all(step_sliced_data == expected_step_slice), (
        f"Step slicing failed for {uintx_tensor.dtype} on {uintx_tensor.device}\n"
        f"Original tensor: {original_data}\n"
        f"Expected step slice: {expected_step_slice}\n"
        f"Actual step slice: {step_sliced_data}"
    )    
    assert step_sliced_data.shape == expected_step_slice.shape, (
        f"Shape mismatch for {uintx_tensor.dtype} on {uintx_tensor.device}\n"
        f"Expected shape: {expected_step_slice.shape}\n"
        f"Actual shape: {step_sliced_data.shape}"
    )

def test_negative_indexing(uintx_tensor_and_dtype):
    uintx_tensor, dtype = uintx_tensor_and_dtype
    negative_sliced_uintx = uintx_tensor[-3:]
    negative_sliced_data = negative_sliced_uintx.get_plain()
    
    original_data = uintx_tensor.get_plain()
    expected_negative_slice = original_data[-3:]
    
    expected_negative_slice = expected_negative_slice.to(negative_sliced_data.dtype)
    
    assert torch.all(negative_sliced_data == expected_negative_slice), (
        f"Negative indexing failed for {uintx_tensor.dtype} on {uintx_tensor.device}\n"
        f"Original tensor: {original_data}\n"
        f"Expected negative slice: {expected_negative_slice}\n"
        f"Actual negative slice: {negative_sliced_data}"
    )
    
    assert negative_sliced_data.shape == expected_negative_slice.shape, (
        f"Shape mismatch for {uintx_tensor.dtype} on {uintx_tensor.device}\n"
        f"Expected shape: {expected_negative_slice.shape}\n"
        f"Actual shape: {negative_sliced_data.shape}"
    )
    
    assert torch.all(negative_sliced_data == original_data[-3:]), (
        f"Negative indexing did not select the correct elements for {uintx_tensor.dtype} on {uintx_tensor.device}\n"
        f"Expected last three elements: {original_data[-3:]}\n"
        f"Actual selected elements: {negative_sliced_data}"
    )

def test_slice_assignment(uintx_tensor_and_dtype):
    uintx_tensor, original_dtype = uintx_tensor_and_dtype
    assert original_dtype in _DTYPE_TO_BIT_WIDTH.keys(), f"Unexpected dtype: {original_dtype}"

    #original data
    original_data = uintx_tensor.get_plain()
    print(f"Original data: {original_data}")

    # data to assign
    new_data = torch.tensor([1, 2], dtype=torch.uint8, device=uintx_tensor.device)
    print(f"New data: {new_data}")

    # quantize the new data to the original dtype
    quantized_new_data = torch.tensor([quantize_for_dtype(v.item(), original_dtype) for v in new_data],
                                      dtype=torch.uint8, device=uintx_tensor.device)

    # assign the quantized data to the slice
    uintx_tensor[3:5] = to_uintx(quantized_new_data, original_dtype)

    # Get the modified data
    modified_data = uintx_tensor.get_plain()
    print(f"Modified data: {modified_data}")

    # Check if the assigned slice has been updated
    assert torch.all(modified_data[3:5] == quantized_new_data), (
        f"Slice assignment failed for {original_dtype} on {uintx_tensor.device}\n"
        f"Assigned slice: {quantized_new_data}\n"
        f"Expected quantized slice: {quantized_new_data}\n"
        f"Actual slice after assignment: {modified_data[3:5]}"
    )

    # Check if the rest of the tensor remained unchanged
    assert torch.all(modified_data[:3] == original_data[:3]) and torch.all(modified_data[5:] == original_data[5:]), (
        f"Unassigned parts of the tensor changed after slice assignment for {original_dtype} on {uintx_tensor.device}"
    )

    # Test assigning a regular tensor (not UintxTensor)
    regular_tensor = torch.tensor([3, 1], dtype=torch.uint8, device=uintx_tensor.device)
    quantized_regular_tensor = torch.tensor([quantize_for_dtype(v.item(), original_dtype) for v in regular_tensor],
                                            dtype=torch.uint8, device=uintx_tensor.device)
    uintx_tensor[5:7] = to_uintx(quantized_regular_tensor, original_dtype)

    modified_data_2 = uintx_tensor.get_plain()

    assert torch.all(modified_data_2[5:7] == quantized_regular_tensor), (
        f"Slice assignment with regular tensor failed for {original_dtype} on {uintx_tensor.device}\n"
        f"Assigned slice: {quantized_regular_tensor}\n"
        f"Expected quantized slice: {quantized_regular_tensor}\n"
        f"Actual slice after assignment: {modified_data_2[5:7]}"
    )

    # Test assigning a scalar value
    scalar_value = 2
    quantized_scalar = quantize_for_dtype(scalar_value, original_dtype)
    uintx_tensor[7] = quantized_scalar

    modified_data_3 = uintx_tensor.get_plain()

    assert modified_data_3[7] == quantized_scalar, (
        f"Scalar assignment failed for {original_dtype} on {uintx_tensor.device}\n"
        f"Assigned scalar: {quantized_scalar}\n"
        f"Expected quantized scalar: {quantized_scalar}\n"
        f"Actual value after assignment: {modified_data_3[7]}"
    )

    print(f"Slice and scalar assignment tests passed for {original_dtype} on {uintx_tensor.device}")

def test_out_of_bounds_slicing(uintx_tensor_and_dtype):
    uintx_tensor, original_dtype = uintx_tensor_and_dtype
    out_of_bounds_uintx = uintx_tensor[5:10]
    out_of_bounds_data = out_of_bounds_uintx.get_plain()
    
    original_data = uintx_tensor.get_plain()
    expected_out_of_bounds = original_data[5:]
    
    assert torch.all(out_of_bounds_data == expected_out_of_bounds), (
        f"Out of bounds slicing failed for {original_dtype} on {uintx_tensor.device}\n"
        f"Original tensor: {original_data}\n"
        f"Expected out of bounds slice: {expected_out_of_bounds}\n"
        f"Actual out of bounds slice: {out_of_bounds_data}"
    )
    
    assert out_of_bounds_data.shape == expected_out_of_bounds.shape, (
        f"Shape mismatch for out of bounds slicing with {original_dtype} on {uintx_tensor.device}\n"
        f"Expected shape: {expected_out_of_bounds.shape}\n"
        f"Actual shape: {out_of_bounds_data.shape}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_transfer(uintx_tensor_and_dtype):
    uintx_tensor_cpu, original_dtype = uintx_tensor_and_dtype
    uintx_tensor_cuda = uintx_tensor_cpu.to("cuda")
    
    assert uintx_tensor_cuda.device.type == "cuda", (
        f"Failed to transfer {original_dtype} tensor to CUDA"
    )
    
    cpu_data = uintx_tensor_cpu.get_plain()
    cuda_data = uintx_tensor_cuda.cpu().get_plain()
    
    assert torch.all(cpu_data == cuda_data), (
        f"Data mismatch after device transfer for {original_dtype}\n"
        f"CPU data: {cpu_data}\n"
        f"CUDA data (transferred back to CPU): {cuda_data}"
    )