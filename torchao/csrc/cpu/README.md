# CPU kernels

CPU kernels are contained in 3 directories:

* torch_free_kernels: This directory contains CPU kernels written with raw pointers and do not use any PyTorch concepts like Tensor.

* shared_kernels: This directory is for kernels that are shared between PyTorch/ATen and Executorch.  They can be compiled with either platform using compile flags.  Kernels in this directory often use torch_free_kernels in their implementation.

* aten_kernels: This directory is for kernels written for PyTorch/ATen.

If possible, we prefer contributors write a shared kernel when constributing new code.
