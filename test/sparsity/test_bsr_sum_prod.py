import torch

import triton
import triton.language as tl
import pdb

from torchao.sparsity.utils import create_block_sparse_tensor
from torchao.sparsity.blocksparse import BlockSparseTensor
from torch.library import wrap_triton, triton_op



@torch.compile(dynamic=False, fullgraph=True)
def test(w, x):
    b = x.unsqueeze(0)
    out= (torch.mul(w, b)).sum(dim=1)
    return out

torch.set_printoptions(profile='full', linewidth=100000)
torch.manual_seed(0)
size = 98432

with torch.no_grad():
    create_block_sparse_tensor = torch.compiler.disable(create_block_sparse_tensor)
    a = create_block_sparse_tensor(32, 32, 16, 0.5, torch.bfloat16).cuda() * torch.randn(32, 32, dtype=torch.bfloat16).cuda()
    a[:16, :16] *= 4
    a[16:, 16:] *= 4
    a[16:, :16] *= 2
    a[:16, 16:] *= 1
    # print(a)
    # print(x)
    w = BlockSparseTensor.from_dense(a, 16).detach()
    x = torch.arange(32).reshape((32, 1)).to(torch.bfloat16).cuda()   
    # expected= test(a.unsqueeze(2), x)
    # print(expected)
    # print("strides", w.unsqueeze(2).stride())
    # print("strides", w.stride())
    out = test(w.unsqueeze(2), x)
    # print(out)

    # torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
