# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from torchao.utils import TorchAOBaseTensor

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
    """The Layout class serves as a base class for defining different data layouts for tensors.
    It provides methods for pre-processing and post-processing tensors, as well as static
    pre-processing with additional parameters like scale, zero_point, and block_size.

    The Layout class is designed to be extended by other layout classes that define specific
    data representations and behaviors for tensors. It is used in conjunction with TensorImpl
    classes to represent custom data layouts and how tensors interact with different operators.
    """

    def pre_process(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def post_process(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, scale, zero_point

    def pre_process_static(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pre_process(input), scale, zero_point

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        return ""


@dataclass(frozen=True)
class PlainLayout(Layout):
    """PlainLayout is the most basic layout class, inheriting from the Layout base class.
    It does not add any additional metadata or processing steps to the tensor.
    Typically, this layout is used as the default when no specific layout is required.
    """

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


###############################
# Base Tensor Impl Subclass #
###############################
class AQTTensorImpl(TorchAOBaseTensor):
    """
    Base class for the tensor impl for `AffineQuantizedTensor`

    Note: This is not a user facing API, it's used by AffineQuantizedTensor to construct
    the underlying implementation of a AQT based on layout
    """

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get the plain (unpacked) Tensor for the tensor impl

        Returns data, scale and zero_point
        Can be overwritten if other types of AQTTensorImpl has different numbers of plain tensors
        """
        pass

    def get_layout(self) -> Layout:
        pass

    @classmethod
    def from_plain(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """Construct a TensorImpl from data, scale, zero_point and the _layout"""
        pass

    def __repr__(self):
        data, scale, zero_point = self.get_plain()
        _layout = self.get_layout()
        return f"{self.__class__.__name__}(data={str(data)}... , scale={str(scale)}... , zero_point={str(zero_point)}... , _layout={_layout})"
