import torch

from triton.testing import do_bench
from torchao.sparsity.utils import create_binary_tensor
from torchao.kernel.splitk_sparse_gemv import splitk_sparse_gemv
import torch.nn.functional as F


for sparsity_level in [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 0.9, 0.95]:

    a = create_binary_tensor((1, 1, 4096), sparsity_level).cuda().to(torch.float16)
    b = torch.randn(16384, 4096).cuda().to(torch.float16).T.contiguous().T

    sparse_time = do_bench(lambda: splitk_sparse_gemv(a, b)) * 1e6
    dense_time = do_bench(lambda: F.linear(a, b)) * 1e6
    speedup = dense_time / sparse_time
    print(f"sparsity_level: {sparsity_level:.2f} | sparse time: {sparse_time:.2f} | dense_time: {dense_time:.2f} | speedup: {speedup:.2f}")
