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

### Benchmarking (Performance)

For ViT-L we see the following e2e speedups on a single NVIDIA A100 across a single training (forwards + backwards) pass with torch.compile enabled and FP16 dtype:


To reproduce these benchmarks, please run:
```
pip install segment-anything-fast pandas
python benchmarks/benchmark_semi_structured_training.py
```

If you have existing matmul shapes for your nn.Linear layers and are curious about the potential speedups, you can run add your shapes [here]() and run microbenchmarks with:
```
python benchmarks/benchmark_semi_structured_training.py --linear
```

### Benchmarking (Accuracy)

When combined with [DINOv2](https://github.com/facebookresearch/dinov2), we found that we were able to train an ImageNet classifier with minimal accuracy loss.

A fully sparse 2:4 trained model exhibited a -0.5 pp accuracy drop; we were able to further reduce the accuracy loss to -0.1 pp by first training with 2:4 sparsity enabled and then switching over to normal dense training.

| Training Configuration                 | Accuracy (%)    |
|----------------------------------------|-----------------|
| 0% Sparse: 125k dense steps (baseline)            | 82.8 |
| 40% Sparse: 40k sparse -> 85k dense steps         | 82.9 |
| 60% Sparse: 75k sparse -> 50k dense steps         | 82.8 |
| 70% Sparse: 87.5k sparse -> 37.5k dense steps     | 82.7 |
| 80% Sparse: 100k sparse -> 25k dense steps        | 82.7 |
| 90% Sparse: 112.5k sparse -> 12.5k dense steps    | 82.0 |
| 100% Sparse: 125k sparse steps (2:4-sparse model) | 82.3 |

All our experiments were run on 4x AMD EPYC 7742 64-core CPUs and 4x NVIDIA A100-80GB GPUs.
