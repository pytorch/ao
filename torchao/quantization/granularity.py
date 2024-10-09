# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(frozen=True)
class Granularity:
    """
    Base class for representing the granularity of quantization.

    This class serves as a parent for specific granularity types used in 
    quantization operations, such as per-tensor or per-axis quantization.
    """
    pass

@dataclass(frozen=True)
class PerTensor(Granularity):
    """
    Represents per-tensor granularity in quantization.

    This granularity type calcualtes the quantization parameters
    based off the entire tensor.
    """
    pass

@dataclass(frozen=True)
class PerAxis(Granularity):
    """
    Represents per-axis granularity in quantization.

    This granularity type calcualtes different quantization parameters
    along a specified axis of the tensor.

    For example if the input tensor is shape [8, 16] and axis=0, then
    the quantization parameters are calculated for each row of the tensor.
    Giving a total of 8 quantization parameters.


    Attributes:
        axis (int): The axis along which reduction is performed.
    """
    axis: int

@dataclass(frozen=True)

class PerGroup(Granularity):
    """
    Represents per-channel group granularity in quantization.

    This granularity type calcualtes different quantization parameters
    for each group of <group_size> elements.

    For example if the input tensor is shape [8, 16], and the group size is 4, then
    the input tensor is reshaped to [64, 4]
    quantization parameters are calculated for each group of 4 elements,
    giving a total of 64 quantization parameters.

    Attributes:
        group_size (int): The size of each quantization group

    """
    group_size: int

class PerRow(Granularity):
    """
    Represents row-wise granularity in quantization.

    This is a special case of per-axis quantization and is unique to Float8 matmuls
    where the input is quantized with a block_size of (1, ..., input.shape[-1]). And the weight
    is quantized with a block_size of (1, weight.shape[1]).
    """
    pass
