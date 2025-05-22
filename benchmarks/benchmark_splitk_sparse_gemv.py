import torch
import torch.nn.functional as F
from triton.testing import do_bench

from torchao.kernel.splitk_sparse_gemv import splitk_sparse_gemv
from torchao.sparsity.utils import create_binary_tensor

dtype = torch.float8_e4m3fn


for sparsity_level in [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 0.9, 0.95]:
    a = create_binary_tensor((1, 4096), sparsity_level).cuda().to(dtype)
    b = torch.randn(16384, 4096).cuda().to(dtype).T.contiguous().T

    sparse_time = (
        do_bench(lambda: splitk_sparse_gemv(a, b, out_dtype=torch.bfloat16)) * 1e6
    )

    dense_time = (
        do_bench(lambda: F.linear(a.to(torch.float16), b.to(torch.float16))) * 1e6
    )
    # b = torch.randn(4096, 16384).cuda().to(dtype).T.contiguous().T
    # dense_time = do_bench(lambda: torch._scaled_mm(a.squeeze(0), b,
    #                                                scale_a=torch.Tensor([1]).cuda(),
    #                                                scale_b=torch.Tensor([1]).cuda(),
    #                                                out_dtype=torch.bfloat16)) * 1e6
    speedup = dense_time / sparse_time
    print(
        f"sparsity_level: {sparsity_level:.2f} | sparse time: {sparse_time:.2f} | dense_time: {dense_time:.2f} | speedup: {speedup:.2f}"
    )
