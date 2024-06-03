# Training acceleration for sparsity

This folder contains a prototype implementation of fast sparse training, utilizing the runtime semi-structured (2:4) sparsification routines present in core.

### Quickstart
```python
import torch
from torchao.sparsity.prototype.training import (
    SemiSparseLinear,
    SemiSparseActivationLinear,
    swap_linear_with_semi_sparse_linear_,
    swap_semi_sparse_linear_with_linear_,
)

model = torch.nn.Sequential(torch.nn.Linear(64, 64)).cuda().to(torch.bfloat16)

sparse_config = {
    "seq.0": SemiSparseLinear,
    # for activation sparsity, uncomment the below line
    #"seq.0": SemiSparseActivationLinear,
}

# swap linear with semi_sparse linear, after this you can train your model as usual.
swap_linear_with_semi_sparse_linear_(model, sparse_config)

# now you can run your normal training loop

# if you need to swap back from semi_sparse linear to normal linear, we provide a utility function
swap_semi_sparse_linear_with_linear(model)
```

### Benchmarking

If you want to see the expected speedups of applying runtime semi-structured sparsity, you can do so by modifying the existing benchmark code in to add your matmul shapes in:
`benchmarks/benchamrk_semi_sparse.py`

```
python benchmarks/benchmark_semi_sparse.py
```

For VIT-L MLP shapes we see the following results:
```
[------------------------------------------------ mlpfwbw -------------------------------------------------]
                                  |   act24   |   dense   |   w24    |  s24_inp_sparsify24  |  s24_inp_clone
1 threads: -------------------------------------------------------------------------------------------------
      f16 (44160,1024,4096,1024)  |  11881.0  |  11534.3  |  9204.7  |        255.1         |      125.8

Times are in microseconds (us).
```

For more information about our API and how it works, please see our blog post.
