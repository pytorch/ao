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

    This granularity type calculates the quantization parameters
    based off the entire tensor.
    """

    pass


@dataclass(frozen=True)
class PerAxis(Granularity):
    """
    Represents per-axis granularity in quantization.

    This granularity type calculates different quantization parameters
    along a specified axis of the tensor.

    Examples:
    * input_tensor shape [A, B], axis 0 -> scale_shape [A, 1]
    * input_tensor shape [A, B], axis 1 -> scale_shape [1, B]
    * input_tensor shape [A, B, C], axis 1 -> scale_shape [1, B, 1]

    Attributes:
        axis (int): The axis which is kept, reduction is performed across all
          the other axes
    """

    axis: int


@dataclass(frozen=True)
class PerGroup(Granularity):
    """
    Represents per-channel group granularity in quantization.

    This granularity type calculates different quantization parameters
    for each group of <group_size> elements.

    For example if the input tensor is shape [8, 16], and the group size is 4, then
    the input tensor is reshaped to [64, 4]
    quantization parameters are calculated for each group of 4 elements,
    giving a total of 64 quantization parameters.

    Attributes:
        group_size (int): The size of each quantization group

    """

    group_size: int


@dataclass(frozen=True)
class PerRow(Granularity):
    """
    Represents row-wise granularity in quantization.

    For 2D tensors, this is a special case of per-axis quantization and is unique to Float8 matmuls
    where the input is quantized with a block_size of (1, ..., input.shape[-1]). And the weight
    is quantized with a block_size of (1, weight.shape[1]).

    TODO(before land): modify docblock for new axis argument
    """

    # TODO(before land): any BC concerns with loading old checkpoints
    # serialized without this arg? investigate this
    dim: int = -1


@dataclass(frozen=True)
class PerToken(Granularity):
    """
    Represents per-token granularity in quantization.

    This granularity type calculates a different set of quantization parameters
    for each token, which is represented as the last dimension of the tensor.

    For example, if the input tensor has shape [2, 3, 4], then there are 6 tokens
    with 4 elements each, and we will calculate 6 sets of quantization parameters,
    one for each token.

    If the input tensor has only two dimensions, e.g. [8, 16], then this is
    equivalent to `PerAxis(axis=0)`, which yields 8 sets of quantization parameters.
    """

    pass


@dataclass(frozen=True)
class PerBlock(Granularity):
    """
    Represents multidimensional per-block granularity in quantization.

    Example:
    * block_size has shape [X, Y]
    * input_tensor shape [A] -> scaling undefined
    * input_tensor shape [A, B] -> scale shape [A // X, B // Y]
    * input_tensor shape [A, B, C] -> scale shape [A, B // X, C // Y]
    * input_tensor shape [A, B, C, D] -> scale shape [A, B, C // X, D // Y], and so on

    Note that `PerBlock((1, Y))` is equivalent to `PerGroup(Y)`

    Attributes:
        block_size (tuple[int, ...]): The size of each quantization group
    """

    # TODO(future PR): consider renaming this attribute to make the meaning
    #   of `block_size` consistent.
    # 1. `block_size` in this class can support tensors of multiple ranks
    # 2. `block_size` in other places in the codebase has rank equal to the
    #    corresponding tensor
    # TODO(future PR): change to list or support serialization with tuples,
    # currently serialization only works when `block_size` is specified as a
    # list. Example error:
    # https://gist.github.com/vkuzo/ab4d6aec83cb98ad9417898d2c024a2c
    block_size: tuple[int, ...]
