# Torch free kernels

Kernels in this directory do not depend on Torch.  Rather than use Tensor, they are written with raw pointers.  These raw kernels are used by ATen/ExecuTorch kernels in torchao/csrc/cpu/shared_kernels.

Code is organized into subdirectories by CPU architecture:
* aarch64 (Arm)
* fallback (architecture-independent / generic C++)
* interface (high-level interface for fallback and architecture-specific code)
