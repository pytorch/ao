# Accelerated Sparse Training

This folder contains an implementation of accelerated sparse training.
<!--For more information about our API and how it works, please see our blog post. (Will add link when its public)-->

Special thanks to @danthe3rd for writing the runtime semi-structured (2:4) sparsification [kernels](https://github.com/pytorch/pytorch/pull/122350) in core.

### Quickstart

**NOTE: This feature is only available on the pytorch / torchao nightlies currently and requires CUDA compute capability 8.0+**

```python
import torch
from torchao.sparsity.training import (
    SemiSparseLinear,
    SemiSparseActivationLinear,
    swap_linear_with_semi_sparse_linear,
    swap_semi_sparse_linear_with_linear,
)

model = torch.nn.Sequential(torch.nn.Linear(1024, 4096)).cuda().to(torch.float16)

# Specify the fully-qualified-name of the nn.Linear modules you want to swap
sparse_config = {
    "seq.0": SemiSparseLinear,
    # for activation sparsity, uncomment the below line
    # "seq.0": SemiSparseActivationLinear,
}

# For DINO ViT training we found that sparsifying the Linear layers of the MLP block only
# to be an acceptable configuration, but the optimal configuration depends on your specific
# model architecture.

# Swap nn.Linear with SemiSparseLinear
swap_linear_with_semi_sparse_linear(model, sparse_config)

# Now you can run your normal training loop

# If you need to swap back from semi_sparse linear to normal linear, we provide a utility function to do so
swap_semi_sparse_linear_with_linear(model)
```

### Benchmarking

If you want to see the expected speedups of applying runtime semi-structured sparsity for training, you can do so by modifying the existing benchmark code in to add your matmul shapes in:
[benchmarks/benchamrk_semi_sparse.py](https://github.com/pytorch/ao/blob/main/benchmarks/benchmark_semi_sparse.py#L25)

```
python benchmarks/benchmark_semi_sparse.py
```

For VIT-L MLP shapes on a NVIDIA A100 we see the following results:
```
[------------------------------------------------ mlpfwbw -------------------------------------------------]
                                  |   act24   |   dense   |   w24    |  s24_inp_sparsify24  |  s24_inp_clone
1 threads: -------------------------------------------------------------------------------------------------
      f16 (44160,1024,4096,1024)  |  11881.0  |  11534.3  |  9204.7  |        255.1         |      125.8

Times are in microseconds (us).
```
