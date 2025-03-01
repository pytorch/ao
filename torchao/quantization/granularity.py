from typing import Annotated, Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


class GranularityBase(BaseModel):
    """Base class for representing the granularity of quantization."""

    model_config = ConfigDict(frozen=True)


class PerTensor(GranularityBase):
    """
    Represents per-tensor granularity in quantization.

    This granularity type calculates the quantization parameters
    based off the entire tensor.
    """

    type: Literal["PerTensor"] = "PerTensor"


class PerAxis(GranularityBase):
    """
    Represents per-axis granularity in quantization.

    This granularity type calculates different quantization parameters
    along a specified axis of the tensor.

    For example if the input tensor is shape [8, 16] and axis=0, then
    the quantization parameters are calculated for each row of the tensor.
    Giving a total of 8 quantization parameters.

    Attributes:
        axis (int): The axis along which reduction is performed.
    """

    type: Literal["PerAxis"] = "PerAxis"
    axis: int


class PerGroup(GranularityBase):
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

    type: Literal["PerGroup"] = "PerGroup"
    group_size: int


class PerRow(GranularityBase):
    """
    Represents row-wise granularity in quantization.

    This is a special case of per-axis quantization and is unique to Float8 matmuls
    where the input is quantized with a block_size of (1, ..., input.shape[-1]). And the weight
    is quantized with a block_size of (1, weight.shape[1]).
    """

    type: Literal["PerRow"] = "PerRow"


class PerToken(GranularityBase):
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

    type: Literal["PerToken"] = "PerToken"


# Create a discriminated union of all granularity types
Granularity = Annotated[
    Union[PerTensor, PerAxis, PerGroup, PerRow, PerToken], Field(discriminator="type")
]


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals(
        [
            set,
        ]
    )
