# Shared kernels

This directory is for kernels that are shared between PyTorch/ATen and Executorch.
Shared kernels are written with abstractions in internal/library.h.
These are compiled to either an ATen or ExecuTorch kernel based on compile flags.
