
import torch
import torchao

A = torch.ones(128, 128).cuda().to(torch.bfloat16)
res = torch.ops.torchao.sparse_semi_structured_tile.default(A, "", False)
print(res)
