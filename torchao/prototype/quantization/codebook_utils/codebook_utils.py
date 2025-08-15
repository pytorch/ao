# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# core ml support scale..
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from torchao.prototype.quantization.codebook.codebook_ops import (
    choose_qparams_codebook,
    dequantize_codebook,
    quantize_codebook,
)
from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
    choose_qparams_and_quantize_codebook_coreml,
)
from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
    dequantize_codebook as dequantize_codebook_coreml,
)
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH


def block_shape_to_group_size(block_shape, tensor_shape):
    """Calculates the total number of elements in a group from a block_shape."""
    n_group, k_group = block_shape
    n_dim, k_dim = tensor_shape

    if n_group == -1:
        n_group = n_dim
    if k_group == -1:
        k_group = k_dim

    return n_group * k_group


def group_size_to_block_shapes(
    lut_group_size: int,
    tensor_shape: Tuple[int, int],
) -> Tuple[List[int], Optional[List[int]]]:
    """
    Translates legacy integer-based group sizes into the new block_shape list format.

    This function encodes the implicit assumptions of the old system:
    - LUTs were always grouped by rows.
    - Scales were always grouped by columns.

    Args:
        lut_group_size (int): The total number of elements that shared a single LUT.
        tensor_shape (Tuple[int, int]): The shape of the weight tensor (N, K).
            This is required to calculate the number of rows for the LUT group.

    Returns:
        A tuple containing:
        - lut_block_shape (List[int]): The new block shape for LUTs (e.g., [N, -1]).
        - scale_block_shape (Optional[List[int]]): The new block shape for scales
          (e.g., [-1, K]), or None.
    """
    n_rows, k_cols = tensor_shape

    # --- 1. Translate LUT Group Size ---
    if lut_group_size % k_cols != 0:
        raise ValueError(
            f"lut_group_size ({lut_group_size}) must be divisible by the number "
            f"of columns ({k_cols}) for legacy row-grouping."
        )
    rows_per_lut = lut_group_size // k_cols
    lut_block_shape = [rows_per_lut, -1]

    return lut_block_shape


@torch.no_grad()
def _quantize_row_wise_group_with_scales(
    input_tensor: torch.Tensor,
    rows_per_group: int,
    scale_block_shape: List[int],
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
            scale_block_size=scale_block_shape[-1],
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
def quantize_flexible_grouping(
    input_tensor: torch.Tensor,
    lut_block_shape: List[int],
    code_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, None]:
    """
    Quantizes a tensor using either row-wise or column-wise grouping.

    Args:
        input_tensor (torch.Tensor): The 2D tensor to be quantized.
          Shape: (n_rows, k_cols)
        lut_block_shape (List[int]): Defines the grouping strategy.
            - To group by columns: `[-1, k_group]`.
            - To group by rows: `[n_group, -1]`.
        code_dtype (torch.dtype): The dtype for the codes (e.g., torch.uint4).

    Returns:
        A tuple containing the quantized codes, the lookup tables, and None.
        - final_codes (torch.Tensor): Quantized data of shape (n_rows, k_cols).
        - final_luts (torch.Tensor): The codebook of lookup tables.
          Shape: (num_groups, 2**nbits), where num_groups depends on the strategy.
        - None: Placeholder for scales, which are not computed.
    """
    assert input_tensor.ndim == 2, "This function expects a 2D tensor."
    assert len(lut_block_shape) == 2, (
        "lut_block_shape must have two elements for a 2D tensor."
    )
    n_rows, k_cols = input_tensor.shape
    n_group, k_group = lut_block_shape

    # STRATEGY 1: Group by ROWS (e.g., block_size = [2, -1])
    if n_group != -1 and k_group == -1:
        assert n_rows % n_group == 0, (
            f"Tensor rows ({n_rows}) must be divisible by row group size ({n_group})."
        )
        list_of_luts, list_of_codes = [], []
        for i in range(0, n_rows, n_group):
            tensor_slice = input_tensor[i : i + n_group, :]
            lut, codes = choose_qparams_and_quantize_codebook_coreml(
                input_tensor=tensor_slice,
                code_dtype=code_dtype,
                block_size=[-1, -1],
            )
            list_of_luts.append(lut)
            list_of_codes.append(codes)

        # Concatenate and remove singleton dimensions
        final_luts = torch.cat(list_of_luts, dim=0).squeeze()
        final_codes = torch.cat(list_of_codes, dim=0)
        return final_codes, final_luts, None

    # STRATEGY 2: Group by COLUMNS (e.g., block_size = [-1, 64])
    elif n_group == -1:
        if k_group != -1:
            assert k_cols % k_group == 0, (
                f"Tensor cols ({k_cols}) must be divisible by col group size ({k_group})."
            )
        luts, codes = choose_qparams_and_quantize_codebook_coreml(
            input_tensor=input_tensor,
            code_dtype=code_dtype,
            block_size=lut_block_shape,
        )
        # Remove singleton dimensions
        final_luts = luts.squeeze()
        final_codes = codes
        return final_codes, final_luts, None

    # Unsupported strategy
    else:
        raise NotImplementedError(
            f"lut_block_shape pattern '{lut_block_shape}' is not supported."
        )


@torch.no_grad
def dequantize_with_flexible_grouping(
    codes: torch.Tensor,
    luts: torch.Tensor,
    lut_block_shape: List[int],
    code_dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert codes.ndim == 2, "This function expects a 2D codes tensor."
    n_rows, k_cols = codes.shape
    n_group, k_group = lut_block_shape

    # STRATEGY 1: Grouping was by COLUMNS (e.g., block_shape = [-1, 64])
    if n_group == -1:
        return dequantize_codebook_coreml(
            codes=codes,
            codebook=luts,
            code_dtype=code_dtype,
            block_size=lut_block_shape,
            output_dtype=output_dtype,
        )

    # STRATEGY 2: Grouping was by ROWS (e.g., block_shape = [2, -1])
    elif n_group != -1 and k_group == -1:
        assert n_rows % n_group == 0, (
            f"Tensor rows ({n_rows}) must be divisible by row group size ({n_group})."
        )
        num_groups = n_rows // n_group
        dequantized_tensor = torch.empty_like(codes, dtype=output_dtype)

        for i in range(num_groups):
            start_row, end_row = i * n_group, (i + 1) * n_group

            # Get the chunk of codes and the single LUT for that chunk
            codes_slice = codes[start_row:end_row, :]
            current_lut = luts[i]

            # To dequantize a chunk with a *single* LUT, we tell the primitive
            # that the block_size should cover all columns (k_cols).
            dequant_slice = dequantize_codebook_coreml(
                codes=codes_slice,
                # The primitive expects a 2D LUT of shape (num_luts, ...).
                # Since we have one LUT, we must add a dimension.
                codebook=current_lut.unsqueeze(0),
                code_dtype=code_dtype,
                block_size=[-1, k_cols],
                output_dtype=output_dtype,
            )
            dequantized_tensor[start_row:end_row, :] = dequant_slice
        return dequantized_tensor

    else:
        raise NotImplementedError(
            f"lut_block_shape pattern '{lut_block_shape}' is not supported."
        )


def quantize_dispatch(
    input_tensor: torch.Tensor,
    lut_block_shape: List[int],
    code_dtype: torch.dtype,
    scale_block_shape: Optional[List[int]] = None,  # Make this optional
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Single entry point for quantization that dispatches to the correct backend.

    This function uses lut_block_shape to determine the quantization strategy,
    allowing for flexible grouping by either rows or columns.

    Args:
        input_tensor (torch.Tensor): The 2D tensor to be quantized (N, K).
        lut_block_shape (List[int]): Defines the grouping for the look-up table.
            - To group by N rows: use `[N, -1]`.
            - To group by K columns: use `[-1, K]`.
        code_dtype (torch.dtype): The target dtype for the codes (e.g., torch.uint4).
        scale_block_shape (Optional[List[int]]): Defines grouping for scale factors,
            used only by the 'scale' backend. E.g., `[-1, 64]`. If provided,
            the 'scale' backend is used in "auto" mode. Defaults to None.
        backend (str): The quantization backend to use. Can be "auto", "coreml",
            or "scale". "auto" chooses based on whether `scale_block_shape` is provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple
            containing the (codes, luts, scales). Scales will be None for the
            'coreml' backend.
    """
    # Determine which backend to use based on if scale_block_shape is provided.
    if backend == "auto":
        backend = "scale" if scale_block_shape is not None else "coreml"

    # Dispatch to the appropriate backend implementation
    if backend == "scale":
        if scale_block_shape is None:
            raise ValueError(
                "'scale' backend requires a `scale_block_shape` to be set."
            )

        # The 'scale' backend only supports row-grouping for the LUT.
        # We derive the rows_per_group from the lut_block_shape parameter.
        n_group, k_group = lut_block_shape
        if n_group == -1 or k_group != -1:
            raise ValueError(
                "The 'scale' backend currently only supports row-grouping for LUTs. "
                "Please use a `lut_block_shape` of `[N, -1]`."
            )
        rows_per_lut_group = n_group

        codes, luts, scales = _quantize_row_wise_group_with_scales(
            input_tensor,
            rows_per_lut_group,
            scale_block_shape,
            code_dtype,
        )

    elif backend == "coreml":
        codes, luts, scales = quantize_flexible_grouping(
            input_tensor, lut_block_shape, code_dtype
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    luts = luts.to(torch.float32)
    return codes, luts, scales


def dequantize_dispatch(
    codes: torch.Tensor,
    luts: torch.Tensor,
    scales: Optional[torch.Tensor],
    lut_block_shape: List[int],
    scale_block_shape: Optional[List[int]] = None,
    backend: str = "auto",
    code_dtype: torch.dtype = torch.int4,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Single entry point for dequantization that dispatches to the correct backend.
    (Updated to use flexible block shapes).
    """
    if backend == "auto":
        # Use presence of scales to determine backend
        backend = "scale" if scales is not None else "coreml"

    if backend == "scale":
        # For backward compatibility, derive old integer args from new block shapes
        if scale_block_shape is None:
            raise ValueError("'scale' backend requires a `scale_block_shape`.")

        n_group, k_group = lut_block_shape
        if k_group != -1:
            raise ValueError(
                "Scale dequant backend only supports row-grouped LUTs ([N, -1])."
            )
        rows_per_lut_group = n_group

        scale_n_group, scale_k_group = scale_block_shape
        if scale_n_group != 1:
            raise ValueError(
                "Scale dequant backend only supports col-grouped scales ([1, K])."
            )
        scale_group_size = scale_k_group

        return _dequantize_row_wise_group_with_scales(
            codes,
            luts,
            scales,
            rows_per_lut_group,
            scale_group_size,
            output_dtype=output_dtype,
        )

    elif backend == "coreml":
        # Perform grouping along rows, reshape the [Rows per group, 2**nbits] LUTs
        # to [1, Rows per group, 2**nbits, 1] for the dequantize primitive.
        num_luts = luts.shape[0]
        lut_size = luts.shape[1]
        luts_4d = luts.reshape(num_luts, 1, lut_size, 1)
        return dequantize_codebook_coreml(
            codes,
            luts_4d,
            _DTYPE_TO_BIT_WIDTH[code_dtype],
            lut_block_shape,
            output_dtype=output_dtype,
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
