# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import types
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.quantization.codebook_coreml.codebook_quantized_tensor import (
    CodebookQuantizedTensor,
)
from torchao.prototype.quantization.codebook_groupwise.codebook_quantized_tensor import (
    CodebookQuantizedPackedTensor,
)
from torchao.quantization.transform_module import register_quantize_module_handler


def _get_linear_extra_repr_for_lut(self) -> str:
    """
    Custom __repr__ for a linear module quantized with GroupwiseLutQuantizedTensor.
    """
    out_features, in_features = self.weight.shape

    # Access metadata from the custom tensor
    bit_width = self.weight.bit_width
    lut_group_size = self.weight.lut_group_size
    scale_group_size = self.weight.scale_group_size

    # The original bias is fused into the packed weight, so self.bias is None.
    has_bias = self.bias is not None

    return (
        f"in_features={in_features}, out_features={out_features}, bias={has_bias}, "
        f"quant=GroupwiseLut(bit_width={bit_width}, lut_gs={lut_group_size}, "
        f"scale_gs={scale_group_size}')"
    )


@dataclass
class GroupwiseLutWeightConfig(AOBaseConfig):
    """
    The primary configuration for groupwise Look-Up Table (LUT) quantization.

    This config uses a `block_shape` to define the quantization strategy,
    allowing for flexible grouping by either rows or columns.

    Args:
        code_dtype (torch.dtype): The target logical dtype for the LUT indices
            (e.g., torch.uint4, torch.int4). This determines the codebook size.
        weight_dtype (torch.dtype): The target dtype for the raw weight (e.g., torch.float32).

        lut_block_shape (List[int]): Defines the grouping for the look-up table.
            This is the key parameter for controlling quantization granularity.
            - To group by N rows: use `[N, -1]`. Example: `[2, -1]` means
              every 2 rows share a single LUT.
            - To group by K columns: use `[-1, K]`. Example: `[-1, 64]` means
              every 64 columns share a single LUT.

        scale_block_shape (Optional[List[int]]): Defines grouping for scale factors,
            used only by the 'scale' backend. If provided, the 'scale' backend
            is automatically selected. The same `[N, -1]` or `[-1, K]` pattern applies.
        has_scale (bool): Whether to use scale factors. Defaults to False.
        target (str): The backend target for the C++ kernel (e.g., "auto", "aten").
    """

    # --- Attributes ---
    code_dtype: torch.dtype = torch.int4
    weight_dtype: torch.dtype = torch.float32
    backend: str = "auto"

    lut_block_shape: List[int] = field(default_factory=lambda: [2, -1])

    scale_block_shape: Optional[List[int]] = None

    use_qdq_reference: bool = False
    target: Optional[str] = None
    cache_dir: Optional[str] = None
    has_scale: bool = False

    def __post_init__(self):
        """Validate the configuration after initialization."""
        # 1. Validate backend string
        if self.backend not in ["auto", "scale", "coreml"]:
            raise ValueError(f"Invalid backend: {self.backend}")

        # 2. Validate lut_block_shape
        if not (
            isinstance(self.lut_block_shape, list) and len(self.lut_block_shape) == 2
        ):
            raise ValueError(
                "`lut_block_shape` must be a list of length 2 (e.g., [N, -1] or [-1, K])."
            )
        if self.lut_block_shape.count(-1) != 1:
            raise ValueError(
                "`lut_block_shape` must contain exactly one '-1' to specify the grouping dimension."
            )

        # 3. Validate scale_block_shape if it exists
        if self.scale_block_shape is not None:
            if not (
                isinstance(self.scale_block_shape, list)
                and len(self.scale_block_shape) == 2
            ):
                raise ValueError(
                    "`scale_block_shape` must be a list of length 2 if provided."
                )


@register_quantize_module_handler(GroupwiseLutWeightConfig)
def _groupwise_lut_weight_transform(
    module: torch.nn.Module, config: GroupwiseLutWeightConfig
) -> torch.nn.Module:
    """
    Transforms a linear module by applying groupwise LUT-based weight quantization.
    Automatically caches results if config.cache_dir is set, using a hash of
    the weight tensor for a unique key.
    """
    assert isinstance(module, torch.nn.Linear), (
        "This transform only applies to torch.nn.Linear modules."
    )
    weight = module.weight.data

    quantized_tensor = CodebookQuantizedTensor.from_float(
        weight, code_dtype=config.code_dtype, block_size=config.lut_block_shape
    )

    if not config.use_qdq_reference:
        packed_weight = CodebookQuantizedPackedTensor.from_codebook_quantized_tensor(
            tensor=quantized_tensor, bias=module.bias
        )
        module.weight = torch.nn.Parameter(packed_weight, requires_grad=False)
        if module.bias is not None:
            module.bias = None
        module.extra_repr = types.MethodType(_get_linear_extra_repr_for_lut, module)

    else:  # For reference, dequantize back to float
        dequantized_weight = quantized_tensor.dequantize(config.weight_dtype)
        module.weight.data.copy_(dequantized_weight)

    return module
