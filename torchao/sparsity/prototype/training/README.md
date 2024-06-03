# Training acceleration for sparsity

This folder contains a prototype implementation of fast sparse training, utilizing the runtime semi-structured (2:4) sparsification routines present in core.

### Quickstart
```
import torch
from torchao.sparsity.prototype.training import SemiSparseLinear, SemiSparseActivationLinear, swap_linear_with_semi_sparse_linear_

model = torch.nn.Sequential(torch.nn.Linear(64, 64)).cuda().to(torch.bfloat16)

sparse_config = {
    "seq.0": SemiSparseLinear,
    # for activation sparsity, uncomment the below line
    "seq.0": SemiSparseActivationLinear,
}

swap_linear_with_semi_sparse_linear_(model, sparse_config)
```

For more information about our API and how it works, please see our blog post.
