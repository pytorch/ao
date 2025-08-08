# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# core ml support scale..
import hashlib
import os
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.quantization.codebook.codebook_ops import (
    choose_qparams_codebook,
    dequantize_codebook,
    quantize_codebook,
)
from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
    choose_qparams_and_quantize_codebook_coreml,
)
from torchao.quantization.granularity import Granularity, PerGroup
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH
from torchao.quantization.transform_module import register_quantize_module_handler

from .codebook_quantized_tensor import GroupwiseLutQuantizedTensor


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

    This single config controls two main quantization recipes:
    1.  ** K-Means (with scales)**:
        This is the recommended, high-accuracy mode. It uses a hierarchical
        grouping where a larger LUT group contains smaller scale groups.

    2.  **CoreML-Style K-Means (no scales)**

    Args:
        weight_dtype (torch.dtype): The target dtype for the LUT indices (e.g., torch.uint4).
        lut_granularity (PerGroup): The group size for the Look-Up Table, the number here mean the exact number of weight inside the single group.
        scale_granularity (Optional[PerGroup]): The group size for scaling factors, the number of exact number of weight inside the single scale group.
        target (str): The backend target for the C++ kernel (e.g., "auto", "aten").
    """

    weight_dtype: torch.dtype = torch.uint4
    lut_granularity: Granularity = PerGroup(128)
    scale_granularity: Optional[Granularity] = PerGroup(64)
    use_qdq_reference: bool = False
    target: Optional[str] = None
    backend: str = "auto"
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        has_scales = self.scale_granularity is not None
        if self.backend not in ["auto", "scale", "coreml"]:
            raise ValueError(f"Invalid backend: {self.backend}")

        if has_scales:
            if not isinstance(self.scale_granularity, PerGroup):
                raise TypeError(
                    f"scale_granularity must be PerGroup, but got {type(self.scale_granularity)}"
                )
            if not isinstance(self.lut_granularity, PerGroup):
                raise TypeError(
                    f"lut_granularity must be PerGroup, but got {type(self.lut_granularity)}"
                )

            # Enforce that the LUT group is larger than or equal to the scale group,
            # and that it is a perfect multiple.
            if self.scale_granularity.group_size > self.lut_granularity.group_size:
                raise ValueError(
                    f"scale_granularity.group_size ({self.scale_granularity.group_size}) cannot be larger than "
                    f"lut_granularity.group_size ({self.lut_granularity.group_size})."
                )
            if self.lut_granularity.group_size % self.scale_granularity.group_size != 0:
                raise ValueError(
                    f"lut_granularity.group_size ({self.lut_granularity.group_size}) must be a multiple of "
                    f"scale_granularity.group_size ({self.scale_granularity.group_size})."
                )


@torch.no_grad()
def _quantize_row_wise_group_with_scales(
    input_tensor: torch.Tensor,
    rows_per_group: int,
    scale_group_size: int,
    code_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantizes a 2D tensor using row-wise grouping, with a unique LUT and
    set of scales for each group.

    Returns a tuple of (codes, luts, scales) with structured shapes.
    """
    assert input_tensor.ndim == 2, "This function expects a 2D tensor."
    n_rows, k_cols = input_tensor.shape
    assert n_rows % rows_per_group == 0, (
        f"Tensor rows ({n_rows}) must be divisible by rows_per_group ({rows_per_group})."
    )

    num_groups = n_rows // rows_per_group
    list_of_luts, list_of_codes, list_of_scales = [], [], []

    for i in range(num_groups):
        start_row = i * rows_per_group
        end_row = start_row + rows_per_group
        tensor_slice = input_tensor[start_row:end_row, :]

        # This performs scalar quantization (block_size=(1, 1)) on the slice
        codebook, scales = choose_qparams_codebook(
            tensor_slice,
            block_size=(1, 1),
            scale_block_size=scale_group_size,
            code_dtype=code_dtype,
        )

        codes = quantize_codebook(
            tensor_slice,
            codebook,
            scales,
            code_dtype=code_dtype,
        )

        # Append results without flattening
        # Squeeze codebook from (codebook_size, 1, 1) to (codebook_size,)
        list_of_luts.append(codebook.squeeze())
        list_of_scales.append(scales)
        list_of_codes.append(codes)

    # Concatenate along the row dimension (dim=0) to preserve structure
    final_codes = torch.cat(list_of_codes, dim=0)
    final_scales = torch.cat(list_of_scales, dim=0)

    # Stack LUTs to create a (num_groups, codebook_size) tensor
    final_luts = torch.stack(list_of_luts, dim=0)
    final_scales = final_scales.flatten()
    return final_codes, final_luts, final_scales


@torch.no_grad()
def _dequantize_row_wise_group_with_scales(
    codes: torch.Tensor,
    luts: torch.Tensor,
    scales: torch.Tensor,
    rows_per_group: int,
    scale_group_size: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes a 2D tensor that was quantized with `quantize_per_row_group_with_scales`.

    Args:
        codes (torch.Tensor): The quantized data codes.
                              Shape: (total_rows, total_cols)
        luts (torch.Tensor): The lookup tables (codebooks) for each group.
                             Shape: (num_groups, codebook_size)
        scales (torch.Tensor): The scale factors for each row.
                               Shape: (total_rows,)
        rows_per_group (int): The number of rows in each quantization group.
        output_dtype (torch.dtype): The desired data type for the output tensor.

    Returns:
        torch.Tensor: The dequantized tensor.
                      Shape: (total_rows, total_cols)
    """
    assert codes.ndim == 2, "This function expects a 2D codes tensor."
    n_rows, k_cols = codes.shape
    assert n_rows % rows_per_group == 0, (
        f"Tensor rows ({n_rows}) must be divisible by rows_per_group ({rows_per_group})."
    )

    # Calculate the number of row groups.
    # e.g., if n_rows=128 and rows_per_group=4, num_groups=32
    num_groups = n_rows // rows_per_group
    assert luts.shape[0] == num_groups, (
        "Mismatch between number of LUTs and row groups."
    )

    # calculate the number of scale blocks per row.
    num_scale_blocks = k_cols // scale_group_size
    # Reshape the flattened scales back to their original 3D structure.
    # Shape: (n_rows, num_scale_blocks, 1)
    reshaped_scales = scales.view(n_rows, num_scale_blocks, 1)

    # Pre-allocate the output tensor for efficiency to avoid creating new tensors in the loop.
    # Shape: (total_rows, total_cols)
    dequantized_tensor = torch.empty_like(codes, dtype=output_dtype)

    # Iterate over each group of rows to dequantize them chunk by chunk.
    for i in range(num_groups):
        # Calculate the start and end row indices for the current group slice.
        start_row = i * rows_per_group
        end_row = start_row + rows_per_group

        # Get the slice of codes for the current group.
        # Shape: (rows_per_group, total_cols), e.g., (4, 64)
        codes_slice = codes[start_row:end_row, :]
        # Get the lookup table (codebook) for the current group.
        # The LUT is 1D, shape: (codebook_size,), e.g., (2,) for 1-bit quantization.
        # Reshape it to the (k, b1, b2) format required by dequantize_codebook.
        # For scalar quantization, block sizes b1 and b2 are 1.
        # Reshaped Shape: (codebook_size, 1, 1), e.g., (2, 1, 1)
        current_lut = luts[i].view(-1, 1, 1)

        # Get the slice of scales corresponding to the rows in this group.
        scales_slice = reshaped_scales[start_row:end_row, :, :]

        # Dequantize the slice using the dedicated function.
        dequant_slice = dequantize_codebook(
            codes=codes_slice,
            codebook=current_lut,
            scales=scales_slice,
            output_dtype=output_dtype,
        )
        # The returned `dequant_slice` has its original shape restored.
        # Shape: (rows_per_group, total_cols), e.g., (4, 64)

        # Place the dequantized slice into the correct position in the final tensor.
        dequantized_tensor[start_row:end_row, :] = dequant_slice

    return dequantized_tensor


@torch.no_grad
def _quantize_row_wise_with_coreml_no_scales(
    input_tensor: torch.Tensor,
    rows_per_group: int,
    code_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, None]:
    """
    Quantizes a tensor by splitting it into groups of rows and calling the
    `coreml` quantization function on each group.

    Args:
        input_tensor (torch.Tensor): The 2D tensor to be quantized.
          Shape: (n_rows, k_cols)
        rows_per_group (int): The number of rows to share a single lookup table.
        code_dtype (torch.dtype): The dtype for the codes (e.g., torch.uint4).

    Returns:
        A tuple containing the quantized codes, the lookup tables, and None.
        - final_codes (torch.Tensor): Quantized data.
          Shape: (n_rows, k_cols)
        - final_luts (torch.Tensor): The codebook of lookup tables.
          Shape: (n_rows // rows_per_group, 2**nbits, 1)
        - None: Placeholder for scales, which are not computed.
    """
    assert input_tensor.ndim == 2, "This function expects a 2D tensor."
    # Get the dimensions of the input tensor.
    # Shape of input_tensor: (n_rows, k_cols)
    n_rows, k_cols = input_tensor.shape
    assert n_rows % rows_per_group == 0, (
        f"Tensor rows ({n_rows}) must be divisible by rows_per_group ({rows_per_group})."
    )

    num_groups = n_rows // rows_per_group
    list_of_luts, list_of_codes = [], []

    # Loop through the tensor in blocks of rows.
    for i in range(num_groups):
        # 1. Get a horizontal slice of the original 2D tensor.
        start_row = i * rows_per_group
        end_row = start_row + rows_per_group
        # Shape of tensor_slice: (rows_per_group, k_cols)
        tensor_slice = input_tensor[start_row:end_row, :]

        # 2. Call the coreml function on the slice. This returns one LUT and the
        #    quantized codes for the current slice. `nbits` is inferred from code_dtype.
        # Shape of lut: (1, 2**nbits, 1)
        # Shape of codes: (rows_per_group, k_cols)
        lut, codes = choose_qparams_and_quantize_codebook_coreml(
            input_tensor=tensor_slice,
            code_dtype=code_dtype,
            block_size=[-1, k_cols],  # Treat all columns as one block
        )

        # 3. Append the results for this group to our lists.
        list_of_luts.append(lut)
        list_of_codes.append(codes)

    # 4. Concatenate all parts into final tensors.
    # We stack the `num_groups` lookup tables along the first dimension.
    # Shape of final_luts: (num_groups, 2**nbits, 1)
    final_luts = torch.cat(list_of_luts, dim=0)

    # We stack the `num_groups` code blocks to reconstruct the full tensor.
    # Shape of final_codes: (num_groups * rows_per_group, k_cols) which is (n_rows, k_cols)
    final_codes = torch.cat(list_of_codes, dim=0)

    return final_codes, final_luts, None


@torch.no_grad
def _dequantize_row_wise_with_coreml_no_scales(
    quantized_codes: torch.Tensor,
    luts: torch.Tensor,
    rows_per_group: int,
    code_dtype: torch.dtype,  # This parameter is no longer needed but kept for signature consistency
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantizes a tensor that was quantized with a row-wise grouping strategy.

    Args:
        quantized_codes (torch.Tensor): The 2D tensor of quantized codes.
          Shape: (n_rows, k_cols)
        luts (torch.Tensor): The codebooks (Look-Up Tables). Must be a 2D tensor
          where each row is a complete lookup table.
          Shape: (n_rows / rows_per_group, 2**nbits)
        rows_per_group (int): The number of rows that share a single lookup table. This must
                              match the value used during quantization.
        code_dtype (torch.dtype): The logical dtype for the codes (e.g., torch.uint4).
        output_dtype (torch.dtype): The desired data type for the output tensor.

    Returns:
        torch.Tensor: The dequantized, reconstructed tensor.
          Shape: (n_rows, k_cols)
    """
    # 1. Validate inputs
    assert quantized_codes.ndim == 2, "This function expects a 2D codes tensor."
    # Shape of quantized_codes: (n_rows, k_cols)
    n_rows, k_cols = quantized_codes.shape
    assert n_rows % rows_per_group == 0, (
        f"Tensor rows ({n_rows}) must be divisible by rows_per_group ({rows_per_group})."
    )

    # The number of groups determines how many lookup tables we should have.
    num_groups = n_rows // rows_per_group
    # Shape of luts: (num_groups, 2**nbits)
    assert luts.ndim == 2, f"LUTs tensor must be 2D, but got {luts.ndim} dimensions."
    assert luts.shape[0] == num_groups, (
        f"Number of LUTs ({luts.shape[0]}) does not match the expected number of groups ({num_groups})."
    )

    # 2. Pre-allocate the output tensor for efficiency
    # Shape of dequantized_tensor: (n_rows, k_cols)
    dequantized_tensor = torch.empty_like(quantized_codes, dtype=output_dtype)

    # 3. Loop through each group of rows and dequantize manually
    for i in range(num_groups):
        # a. Get the slice of codes for the current group.
        start_row = i * rows_per_group
        end_row = start_row + rows_per_group
        # Shape of codes_slice: (rows_per_group, k_cols)
        codes_slice = quantized_codes[start_row:end_row, :]

        # b. Select the single, corresponding lookup table for this group.
        # Shape of current_lut: (2**nbits,)
        current_lut = luts[i]

        # c. Perform the dequantization using advanced indexing.
        # This is the core operation: use the 2D `codes_slice` tensor to look up
        # values in the 1D `current_lut` tensor. PyTorch handles this directly.
        # Shape of dequant_slice: (rows_per_group, k_cols)
        dequant_slice = current_lut[codes_slice]

        # d. Place the dequantized slice into the correct position in the final tensor.
        dequantized_tensor[start_row:end_row, :] = dequant_slice

    return dequantized_tensor


def _quantize_dispatch(
    input_tensor: torch.Tensor,
    rows_per_lut_group: int,
    config: GroupwiseLutWeightConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Single entry point for quantization that dispatches to the correct backend.
    Always returns (codes, luts, scales) for a consistent API.
    """
    # Determine which backend to use, based on if have scales or not
    if config.backend == "auto":
        backend = "scale" if config.scale_granularity else "coreml"
    else:
        backend = config.backend

    # Dispatch to the appropriate backend implementation
    if backend == "scale":
        if not config.scale_granularity:
            raise ValueError(
                "'scale_based' backend requires scale_group_shape to be set."
            )
        codes, luts, scales = _quantize_row_wise_group_with_scales(
            input_tensor,
            rows_per_lut_group,
            config.scale_granularity.group_size,
            config.weight_dtype,
        )
    elif backend == "coreml":
        codes, luts, scales = _quantize_row_wise_with_coreml_no_scales(
            input_tensor, rows_per_lut_group, config.weight_dtype
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    luts = luts.to(torch.float32)
    return codes, luts, scales


def _dequantize_dispatch(
    codes: torch.Tensor,
    luts: torch.Tensor,
    scales: Optional[torch.Tensor],
    rows_per_lut_group: int,
    config: GroupwiseLutWeightConfig,
    scale_group_size: int = -1,
) -> torch.Tensor:
    """
    Single entry point for dequantization that dispatches to the correct backend.
    """
    if config.backend == "auto":
        backend = "scale" if config.scale_granularity else "coreml"
    else:
        backend = config.backend
    if backend == "scale":
        return _dequantize_row_wise_group_with_scales(
            codes, luts, scales, rows_per_lut_group, scale_group_size, torch.float32
        )
    elif backend == "coreml":
        return _dequantize_row_wise_with_coreml_no_scales(
            codes, luts, rows_per_lut_group, config.weight_dtype, torch.float32
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def save_quantized_data(data: Dict[str, Any], filepath: str):
    """
    Saves the dictionary of quantized tensors to a file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(data, filepath)
    print(f"Saved quantization results to '{filepath}'")


def load_quantized_data(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads the dictionary of quantized tensors from a file if it exists.
    """
    if not os.path.exists(filepath):
        return None
    data = torch.load(filepath)
    print(f"Loaded quantization results from cache: '{filepath}'")
    return data


@dataclass
class GroupwiseLutWeightConfig(AOBaseConfig):
    """
    The primary configuration for groupwise Look-Up Table (LUT) quantization.

    This single config controls two main quantization recipes:
    1.  ** K-Means (with scales)**:
        This is the recommended, high-accuracy mode. It uses a hierarchical
        grouping where a larger LUT group contains smaller scale groups.

    2.  **CoreML-Style K-Means (no scales)**

    Args:
        weight_dtype (torch.dtype): The target dtype for the LUT indices (e.g., torch.uint4).
        lut_granularity (PerGroup): The group size for the Look-Up Table. This is the
            exact number of weights that will share a single Look-Up Table.
        scale_granularity (Optional[PerGroup]): The group size for scaling factors. This is the
            exact number of weights that will share a single scale factor.
        target (str): The backend target for the C++ kernel (e.g., "auto", "aten").
    """

    weight_dtype: torch.dtype = torch.uint4
    lut_granularity: Granularity = PerGroup(128)
    scale_granularity: Optional[Granularity] = PerGroup(64)
    use_qdq_reference: bool = False

    # If True, quantizes and then immediately de-quantizes the weight back to
    # float32. Useful for debugging and reference, but does not use custom kernels.
    use_qdq_reference: bool = False

    # Specifies a target for backend-specific C++ kernels (e.g., "aten").
    target: Optional[str] = None

    # Controls the quantization algorithm backend.
    # "auto": Chooses automatically based on whether scales are used.
    # "scale": Enforces the hierarchical algorithm with scaling.
    # "coreml": Enforces the simplified algorithm without scaling.
    backend: str = "auto"
    # Directory to cache the results of the expensive K-Means quantization.
    # Caching is keyed by a hash of the weight tensor and the config.
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Validate the configuration after initialization."""
        has_scales = self.scale_granularity is not None
        if self.backend not in ["auto", "scale", "coreml"]:
            raise ValueError(f"Invalid backend: {self.backend}")

        if has_scales:
            if not isinstance(self.scale_granularity, PerGroup):
                raise TypeError(
                    f"scale_granularity must be PerGroup, but got {type(self.scale_granularity)}"
                )
            if not isinstance(self.lut_granularity, PerGroup):
                raise TypeError(
                    f"lut_granularity must be PerGroup, but got {type(self.lut_granularity)}"
                )

            # Enforce that the LUT group is larger than or equal to the scale group,
            # and that it is a perfect multiple.
            if self.scale_granularity.group_size > self.lut_granularity.group_size:
                raise ValueError(
                    f"scale_granularity.group_size ({self.scale_granularity.group_size}) cannot be larger than "
                    f"lut_granularity.group_size ({self.lut_granularity.group_size})."
                )
            if self.lut_granularity.group_size % self.scale_granularity.group_size != 0:
                raise ValueError(
                    f"lut_granularity.group_size ({self.lut_granularity.group_size}) must be a multiple of "
                    f"scale_granularity.group_size ({self.scale_granularity.group_size})."
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

    # --- Step 1: Caching and Quantization ---
    cache_filepath = None
    if config.cache_dir:
        # Generate a unique key based on weight content and config
        weight_bytes = module.weight.data.cpu().numpy().tobytes()
        weight_hash = hashlib.sha256(weight_bytes).hexdigest()

        dtype_str = str(config.weight_dtype).split(".")[-1]
        lut_gs = config.lut_granularity.group_size
        scale_gs = (
            config.scale_granularity.group_size if config.scale_granularity else "none"
        )
        config_str = (
            f"dtype-{dtype_str}_lut-{lut_gs}_scale-{scale_gs}-backend-{config.backend}"
        )

        hash_prefix = weight_hash[:2]
        filename = f"{weight_hash[2:]}_{config_str}.pt"
        cache_filepath = os.path.join(config.cache_dir, hash_prefix, filename)

    quantized_data = load_quantized_data(cache_filepath) if cache_filepath else None

    if quantized_data is not None:  # Cache HIT
        quantized_weight_indices = quantized_data["codes"]
        luts = quantized_data["luts"]
        scales = quantized_data["scales"]
    else:  # Cache MISS - run the expensive quantization
        print(
            f"Cache miss for weight shape {module.weight.shape}. Running quantization..."
        )
        weight = module.weight.data
        rows_per_lut_group = config.lut_granularity.group_size // weight.shape[1]

        quantized_weight_indices, luts, scales = _quantize_dispatch(
            weight, rows_per_lut_group, config
        )

        # Drop last dimension if it is 1 (scalar quantization)
        if luts.ndim > 1 and luts.shape[-1] == 1:
            luts = torch.squeeze(luts, dim=-1)

        # Save the newly computed results to the cache file
        if cache_filepath:
            data_to_save = {
                "codes": quantized_weight_indices,
                "luts": luts,
                "scales": scales,
            }
            save_quantized_data(data_to_save, cache_filepath)

    # --- Step 2: Replace the module's weight with the quantized format ---
    if not config.use_qdq_reference:
        packed_weight = GroupwiseLutQuantizedTensor.from_packed_data(
            int_data=quantized_weight_indices,
            luts=luts,
            scales=scales,
            bias=module.bias,
            bit_width=_DTYPE_TO_BIT_WIDTH[config.weight_dtype],
            lut_group_size=config.lut_granularity.group_size,
            scale_group_size=config.scale_granularity.group_size
            if config.scale_granularity
            else -1,
            original_shape=module.weight.shape,
            target=config.target,
        )
        module.weight = torch.nn.Parameter(packed_weight, requires_grad=False)
        if module.bias is not None:
            module.bias = None
        module.extra_repr = types.MethodType(_get_linear_extra_repr_for_lut, module)
    else:  # For reference, dequantize back to float
        rows_per_lut_group = config.lut_granularity.group_size // module.weight.shape[1]
        scale_group_size = (
            config.scale_granularity.group_size if config.scale_granularity else -1
        )

        dequantized_weight = _dequantize_dispatch(
            quantized_weight_indices.to(torch.long),
            luts,
            scales,
            rows_per_lut_group,
            config,
            scale_group_size,
        )
        module.weight.data.copy_(dequantized_weight)

    return module
