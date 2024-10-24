import torch
from typing import Union, Tuple
from dataclasses import dataclass

"""
Base class for different layout, following the same design of PyTorch layout
https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout, used to represent different
data layout of a Tensor, it's used in conjunction with TensorImpl to represent custom data layout.

As a native PyTorch example, Sparse Coordinate format Tensor (https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html#torch-sparse-coo-tensor) has `torch.sparse_coo` layout, which is backed up by
`SparseImpl`: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/SparseTensorImpl.h which stores two Tensors (indices_ and values_)

We extended the layout in torchao with Layout class (instead of torch.layout objects), also we use tensor subclass to implement TensorImpl classes.

Layout also allows users to pass around configurations for the TensorImpl,
e.g. inner_k_tiles for int4 tensor core tiled TensorImpl

Note: Layout is an abstraction not only for custom data representation, it is also used for how the
Tensor interacts with different operators, e.g. the same data representation can have different
behaviors when running the same operator, e.g. transpose, quantized_linear. This is the same as layout
in PyTorch native Tensor
"""
@dataclass(frozen=True)
class Layout:
    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def pre_process_static(self, input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, block_size: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pre_process(input), scale, zero_point

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        return ""

"""
Plain Layout, the most basic Layout, also has no extra metadata, will typically be the default
"""
@dataclass(frozen=True)
class PlainLayout(Layout):
    pass

def is_device(target_device_str: str, device: Union[str, torch.device]):
    return torch.device(device).type == target_device_str

def get_out_shape(input_shape: Tuple[int], weight_shape: Tuple[int]) -> Tuple[int, int]:
    """Returns the unflattened shape of the input tensor.
    Args:
        input_shape: The input tensor shape possibly more than 2 dimensions
        weight_shape: The weight tensor shape.
    Returns:
        The unflattened shape of the input tensor.
    """
    out_dim = weight_shape[0]
    inpt_dims = input_shape[:-1]
    return (*inpt_dims, out_dim)
