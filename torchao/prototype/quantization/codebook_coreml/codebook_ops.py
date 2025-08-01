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
        The LUT table has dimension input_tensor.dim() + 2, where:
         * The first input_tensor.dim() dimensions index over the different tables (input_tensor.shape[i] // block_size[i] in each dimension)
         * The input_tensor.dim() + 1 dimension indexes over the nbit indices (2 ** nbits)
         * The input_tensor.dim() + 2 dimension indexes over the look up values (shape = 1 for scalar)
    """
    assert code_dtype in list(_SUB_BYTE_UINT_BOUNDS.keys()) + [torch.uint8]
    nbits = _DTYPE_TO_BIT_WIDTH[code_dtype]
    assert nbits >= 1 and nbits <= 8, f"nbits must be in [1, 8], got {nbits}"

    assert len(block_size) == input_tensor.dim()
    block_size = block_size.copy()
    for i in range(len(block_size)):
        if block_size[i] == -1:
            block_size[i] = input_tensor.shape[i]
        assert block_size[i] >= 1 and input_tensor.shape[i] % block_size[i] == 0, (
            "block_size[i] must divide input_tensor.shape[i]"
        )

    assert input_tensor.dim() == 2, "Currently only rank 2 tensors are supported"
    assert block_size[0] == input_tensor.shape[0], (
        "Currently only support per-grouped channel granularity"
    )
    assert cluster_dim == 1, (
        f"only cluster_dim == 1 is supported right now, got {cluster_dim}"
    )

    num_lut = input_tensor.shape[1] // block_size[1]
    group_size = block_size[1]

    # for converting to numpy
    input_tensor = input_tensor.detach()
    original_shape = input_tensor.shape

    # reshape to (N, K // group_size, group_size)
    input_tensor = input_tensor.reshape(input_tensor.shape[0], num_lut, group_size)
    from coremltools.models.neural_network.quantization_utils import (
        _get_kmeans_lookup_table_and_weight,
    )

    res_lut = []
    # each res_w[:, i, :] will use the same lookup table
    # res_w: (N, K // group_size, group_size)
    res_w = torch.zeros_like(input_tensor, dtype=torch.uint8)
    for i in range(num_lut):
        # lut: (2**nbits, 1)
        # w: (N * group_size)
        lut, w = _get_kmeans_lookup_table_and_weight(
            nbits, input_tensor[:, i, :], force_kmeans1d, cluster_dim, vector_axis
        )
        res_lut.append(torch.from_numpy(lut))
        res_w[:, i, :] = torch.from_numpy(w.reshape(input_tensor.shape[0], group_size))

    # directly stack all lookup tables along dim 0
    # res_lut: (K // group_size, 2 ** nbits)
    res_lut = torch.stack(res_lut, dim=0)

    # The final LUT should have dimension equal to input_tensor.dim() + 2
    # The first input_tensor.dim() dimensions index over the tables,
    # input_tensor.dim() + 1 indexes over the nbit indices
    # input_tensor.dim() + 2 are the look up values (shape = 1 for scalar)
    # res_lut: (N, K // group_size, 2 ** nbits, group_size)
    res_lut = res_lut.reshape(1, num_lut, 2**nbits, 1)

    # reshape back to (N, K)
    res_w = res_w.reshape(*original_shape)

    return res_lut, res_w


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
        block_size (List[int]): a slice of elements with shape block_size will share the same lookup table
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
    ndim = len(code_shape)
    assert len(block_size) == ndim, "block_size must match dimensionality of codes"

    # Compute which codebook slice to use for each element
    group_indices = []
    for dim, bsz in zip(code_shape, block_size):
        assert bsz >= 1 and dim % bsz == 0, (
            f"dimension {dim} not divisible by block size {bsz}"
        )
    for i, bsz in enumerate(block_size):
        indices = torch.arange(code_shape[i], device=codes.device) // bsz
        group_indices.append(indices)

    # Broadcast group_indices to shape of codes
    mesh = torch.meshgrid(*group_indices, indexing="ij")
    group_index_tensor = torch.stack(mesh, dim=-1)  # shape (..., N), where N = ndim

    # Flatten everything to index efficiently
    flat_codes = codes.reshape(-1)
    flat_groups = group_index_tensor.reshape(-1, ndim)  # (..., ndim)

    # Compute dequantized values via indexing
    # index into codebook with (*group_index, code_index, :)
    gathered = codebook[(*flat_groups.T, flat_codes)]  # shape (numel, vec_dim)
    dequant = gathered.reshape(*code_shape, vec_dim)

    if vec_dim == 1:
        dequant = dequant.squeeze(-1)

    return dequant.to(dtype=output_dtype)
