# README

## File Structure of the `dtypes` Folder

The `dtypes` folder contains several important files and subfolders that are organized as follows:

- **affine_quantized_tensor.py**: This is the main file, from which the subfolders `uintx` and `floatx` inherit. It contains the base tensor subclass `AffineQuantizedTensor` and code for layout and tensorImpl registration.

- **affine_quantized_tensor_ops.py**: This file defines all the overriden aten ops and different dispatch kernels related to affine quantized tensors.

- **utils.py**: A utility file that provides helper functions and common utilities used across different files in the `dtypes` folder.

- **nf4tensor.py**: This file is specific to the NF4 tensor implementation, and layouts.

### Subfolders

- **uintx**: A subfolder that contains layouts and tensor subclasses inheriting from `affine_quantized_tensor.py`. It is specialized for handling unsigned integer quantized tensors.

- **floatx**: Similar to `uintx`, this subfolder contains layouts and tensor subclasses that inherit from `affine_quantized_tensor.py`, but it is focused on floating-point quantized tensors.
