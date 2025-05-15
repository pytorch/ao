import torch
from sparse_gemv import splitk_sparse_gemv
from triton.testing import do_bench

def create_binary_tensor(shape, percent_zeros):
    """
    Creates a PyTorch tensor with a specific percentage of zeros and ones.
    
    Args:
        shape (tuple): The shape of the tensor to create
        percent_zeros (float): Percentage of zeros in the tensor (between 0 and 1)
    
    Returns:
        torch.Tensor: A tensor with specified percentage of zeros and ones
    """
    # Calculate the total number of elements
    total_elements = torch.prod(torch.tensor(shape)).item()
    
    # Calculate number of zeros needed
    num_zeros = int(total_elements * percent_zeros)
    
    # Create a vector of all ones
    tensor = torch.ones(total_elements)
    
    # Randomly choose indices to set to zero
    zero_indices = torch.randperm(total_elements)[:num_zeros]
    tensor[zero_indices] = 0
    
    # Reshape to the desired shape
    tensor = tensor.reshape(shape)
    
    return tensor

for sparsity_level in [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 0.9, 0.95]:

    a = create_binary_tensor((1, 1, 4096), sparsity_level).cuda().to(torch.float16)
    b = torch.randn(16384, 4096).cuda().to(torch.float16).T.contiguous().T

    sparse_time = do_bench(lambda: splitk_sparse_gemv(a, b)) * 1e6
    dense_time = do_bench(lambda: torch.matmul(a, b.T)) * 1e6
    speedup = dense_time / sparse_time
    print(f"sparsity_level: {sparsity_level:.2f} | sparse time: {sparse_time:.2f} | dense_time: {dense_time:.2f} | speedup: {speedup:.2f}")
