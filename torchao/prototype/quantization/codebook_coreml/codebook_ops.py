# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional, Tuple

import torch

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _SUB_BYTE_UINT_BOUNDS,
)
from torchao.utils import _register_custom_op

quant_lib = torch.library.Library("quant", "FRAGMENT")
register_custom_op = _register_custom_op(quant_lib)


# wrapper around coreml util: https://github.com/apple/coremltools/blob/1c0e5cb1c1e3ab759af107b54f2be18b7c03f8aa/coremltools/models/neural_network/quantization_utils.py#L363
@torch.no_grad
@register_custom_op
def choose_qparams_and_quantize_codebook_coreml(
    input_tensor: torch.Tensor,
    code_dtype: torch.dtype,
    block_size: List[int],
    force_kmeans1d: bool = False,
    cluster_dim: int = 1,
    vector_axis: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the codebook using k-means clustering on blocks of the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to be quantized.
        code_dtype (torch.dtype): The dtype for the codes. [torch.uint1, ..., torch.uint8]
        block_size (List[int]): block sizes for how many elements in each dimension share
           the same lookup table (len(block_size) == input_tensor.dim())
           Each dimension of input_tensor must be divisible by the corresponding element of block_size
           Look up tables are indexed by {(di // bi) for i in input_tensor.dim()}
           For example, if the input tensor has shape (N, K), and block_size is (N, group_size), this means
           there is a lookup table for group_size columns, i.e., (K // group_size) total look up tables
        force_kmeans1d (bool): Use kmeans1d regardless of number of weights
        cluster_dim (int): this means the size of the vector for vector lookup table quantization
          e.g. when cluster_dim is 4, instead of quantizing each scalar value one by one, we quantize
          the tensor in a unit of 4 element vectors, a vector of original tensor will be mapped to
          a vector in the codebook (lookup table) based on the indices.
        vector_axis (Optional[int]): used in vector quantization, see more docs in https://github.com/apple/coremltools/blob/1c0e5cb1c1e3ab759af107b54f2be18b7c03f8aa/coremltools/optimize/_utils.py#L371

    Returns:
        Tuple[torch.Tensor, torch.Tensor]  The codebook (lookup table) Tensor and the quantized Tensor (codes, torch.uint8)
        The LUT table has dimension (g0, .., g(N-1), 2**nbits, vec_dim), where:
         * The first N dimensions index over the different tables (gi = input_tensor.shape[i] // block_size[i] in each dimension)
         * The N + 1 dimension indexes over the nbit indices (2 ** nbits)
         * The N + 2 dimension indexes over the look up values (shape = 1 for scalar)
    """
    assert code_dtype in list(_SUB_BYTE_UINT_BOUNDS.keys()) + [torch.uint8]
    nbits = _DTYPE_TO_BIT_WIDTH[code_dtype]
    assert nbits >= 1 and nbits <= 8, f"nbits must be in [1, 8], got {nbits}"
    assert input_tensor.dim() == 2, "Currently only rank 2 tensors are supported"
    assert cluster_dim == 1, (
        f"only cluster_dim == 1 is supported right now, got {cluster_dim}"
    )

    original_shape = input_tensor.shape
    N, K = original_shape
    input_tensor = input_tensor.detach()

    # --- Process block_size ---
    assert len(block_size) == 2
    processed_block_size = block_size.copy()
    if processed_block_size[0] == -1:
        processed_block_size[0] = N
    if processed_block_size[1] == -1:
        processed_block_size[1] = K

    row_block_size, col_block_size = processed_block_size
    assert N % row_block_size == 0, (
        f"Tensor rows ({N}) not divisible by row block size ({row_block_size})"
    )
    assert K % col_block_size == 0, (
        f"Tensor cols ({K}) not divisible by col block size ({col_block_size})"
    )

    # --- Determine and execute grouping strategy ---
    assert row_block_size == N or col_block_size == K
    is_col_grouping = row_block_size == N

    res_lut_list = []
    from coremltools.models.neural_network.quantization_utils import (
        _get_kmeans_lookup_table_and_weight,
    )

    if is_col_grouping:
        # STRATEGY 1: Group by COLUMNS
        num_luts = K // col_block_size
        reshaped_tensor = input_tensor.reshape(N, num_luts, col_block_size)
        res_codes = torch.zeros_like(reshaped_tensor, dtype=torch.uint8)

        for i in range(num_luts):
            block_to_quantize = reshaped_tensor[:, i, :]
            lut, w = _get_kmeans_lookup_table_and_weight(
                nbits, block_to_quantize, force_kmeans1d, cluster_dim, vector_axis
            )
            res_lut_list.append(torch.from_numpy(lut))
            res_codes[:, i, :] = torch.from_numpy(w.reshape(N, col_block_size))

        # Shape to match CoreML spec: (1, num_luts, 2**nbits, 1)
        final_luts = torch.stack(res_lut_list, dim=0).reshape(1, num_luts, 2**nbits, 1)

    else:  # is_row_grouping
        # STRATEGY 2: Group by ROWS
        num_luts = N // row_block_size
        reshaped_tensor = input_tensor.reshape(num_luts, row_block_size, K)
        res_codes = torch.zeros_like(reshaped_tensor, dtype=torch.uint8)

        for i in range(num_luts):
            block_to_quantize = reshaped_tensor[i, :, :]
            lut, w = _get_kmeans_lookup_table_and_weight(
                nbits, block_to_quantize, force_kmeans1d, cluster_dim, vector_axis
            )
            res_lut_list.append(torch.from_numpy(lut))
            res_codes[i, :, :] = torch.from_numpy(w.reshape(row_block_size, K))

        final_luts_stacked = torch.stack(
            res_lut_list, dim=0
        )  # Shape: (num_luts, 2**nbits, 1)

        # Reshape to the consistent 4D format
        # The shape is (num_row_groups, 1, 2**nbits, 1)
        final_luts = final_luts_stacked.reshape(num_luts, 1, 2**nbits, 1)

    # Reshape codes back to the original tensor shape
    final_codes = res_codes.reshape(*original_shape)

    return final_luts, final_codes


@register_custom_op
def dequantize_codebook(
    codes: torch.Tensor,
    codebook: torch.Tensor,
    nbits: int,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Reconstructs the original tensor from codes and the codebook.

    Args:
        codes (torch.Tensor): Indices of codebook entries for each element
            General shape: (d0, d1, d2, ..., dN)
            Simple example shape: (N, K)
        codebook (torch.Tensor): Codebook tensor used for quantization
            General shape: (d0 // block_size[0], ..., dN // block_size[N], 2**nbits, vec_dim), where vec_dim = 1 for scalar look up values
            Simple example shape: (1, group_size, 2 ** nbits, 1) for scalar look up values, with 1 table per group_size columns
        nbits: int: number of bits for the quantization
        block_size (List[int]): a slice of elements with shape block_size will share the same lookup table.
            If block_size[i] == -1, then the entire dimension is used.
        output_dtype (torch.dtype): dtype for the output tensor.

    Returns:
        dequant (torch.Tensor): Reconstructed tensor, shape (N, K)

    """
    assert output_dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported output dtype: {output_dtype}"

    assert nbits >= 1 and nbits <= 8, f"nbits must be in [1, 8], got {nbits}"

    assert len(block_size) == codes.dim()
    block_size = block_size.copy()
    for i in range(len(block_size)):
        if block_size[i] == -1:
            block_size[i] = codes.shape[i]
        assert block_size[i] >= 1 and codes.shape[i] % block_size[i] == 0, (
            "block_size[i] must divide codes.shape[i]"
        )

    assert codebook.dim() == codes.dim() + 2
    codebook_shape = codebook.shape
    vec_dim = codebook_shape[-1]
    quant_levels = 2**nbits

    # Check that last two dimensions of codebook are [quant_levels, vec_dim]
    assert codebook_shape[-2] == quant_levels, "Codebook shape mismatch with nbits"

    # Compute shape of lookup group indices from codes shape and block size
    code_shape = codes.shape
    ndim = codes.ndim
    assert len(block_size) == ndim, "block_size must match dimensionality of codes"

    # Compute which codebook slice to use for each element
    group_indices = []
    for i in range(ndim):
        assert block_size[i] >= 1 and code_shape[i] % block_size[i] == 0, (
            f"dimension {code_shape[i]} not divisible by block size {block_size[i]}"
        )

        # Index of block
        idx = (
            torch.arange(code_shape[i], device=codes.device) // block_size[i]
        )  # shape (di,)

        # Reshape idx to broadcast along all other dims
        shape = [1] * ndim
        shape[i] = code_shape[i]
        idx = idx.view(*shape)  # shape (1, ..., 1, di, 1, ..., 1)
        idx = idx.expand(code_shape)  # shape (d0, ..., dN)
        group_indices.append(idx)

    # Stack the broadcasted group indices
    # group_index_tensor at (i0, i1, ..., iN) is the gives the group indices (g0, ..., gN)
    # for the element at (i0, i1, ..., iN) in the original code
    # If code.shape = (d1, d2, d3), then group_index_tensor.shape = (d1, d2, d3, 3)
    group_index_tensor = torch.stack(
        group_indices, dim=-1
    )  # shape (d0, d1, ..., dN, ndim)

    # Flatten everything to index efficiently
    flat_codes = codes.reshape(-1)  # shape (numel,)
    flat_groups = group_index_tensor.reshape(-1, ndim)  # (numel, ndim)

    # Compute dequantized values via indexing
    # index into codebook with (*group_index, code_index, :)
    gathered = codebook[(*flat_groups.T, flat_codes)]  # shape (numel, vec_dim)
    dequant = gathered.reshape(*code_shape, vec_dim)

    if vec_dim == 1:
        dequant = dequant.squeeze(-1)

    return dequant.to(dtype=output_dtype)
