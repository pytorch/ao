import torch
from typing import Union, Tuple
from dataclasses import dataclass

"""
Base class for different LayoutType, should not be instantiated directly
used to allow users to pass around configurations for the layout tensor, e.g. inner_k_tiles
for int4 tensor core tiled layout

Note: layout is an abstraction not only for custom data representation, it is also used for how the
layout interacts with different operators, e.g. the same data representation can have different
behaviors when running the same operator, e.g. transpose, quantized_linear.
"""
@dataclass(frozen=True)
class LayoutType:
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
Plain LayoutType, the most basic LayoutType, also has no extra metadata, will typically be the default
"""
@dataclass(frozen=True)
class PlainLayoutType(LayoutType):
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
