# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
)
from torchao.utils import _register_custom_op

quant_lib = torch.library.Library("quant", "FRAGMENT")
register_custom_op = _register_custom_op(quant_lib)


# wrapper around coreml util: https://github.com/apple/coremltools/blob/1c0e5cb1c1e3ab759af107b54f2be18b7c03f8aa/coremltools/models/neural_network/quantization_utils.py#L363
@torch.no_grad
@register_custom_op
def choose_qparams_and_quantize_codebook(
    input_tensor: torch.Tensor,
    code_dtype: torch.dtype,
    group_size: int,
    force_kmeans1d: bool = False,
    cluster_dim: int = 1,
    vector_axis: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the codebook using k-means clustering on blocks of the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to be quantized.
        code_dtype (torch.dtype): The dtype for the codes.
        group_size (int): the size for how many elements of last dimension of input_tensor
          belong to the same group and should share the same lookup table. let's say original
          shape is (N, K), then the slice of (N, group_size) elements should use the same lookup
          table, and there will be (K // group_size) lookup tables
        force_kmeans1d (bool): Use kmeans1d regardless of number of weights
        cluster_dim (int): this means the size of the vector for vector lookup table quantization
          e.g. when cluster_dim is 4, instead of quantizing each scalar value one by one, we quantize
          the tensor in a unit of 4 element vectors, a vector of original tensor will be mapped to
          a vector in the codebook (lookup table) based on the indices.
        vector_axis (Optional[int]): used in vector quantization, see more docs in https://github.com/apple/coremltools/blob/1c0e5cb1c1e3ab759af107b54f2be18b7c03f8aa/coremltools/optimize/_utils.py#L371

    Returns:
        Tuple[torch.Tensor, torch.Tensor]  The codebook (lookup table) Tensor and the quantized Tensor (codes)
    """
    if group_size == -1:
        group_size = input_tensor.shape[-1]

    assert input_tensor.shape[-1] % group_size == 0
    assert input_tensor.ndim == 2
    assert cluster_dim == 1, (
        f"only cluster_dim == 1 is supported right now, got {cluster_dim}"
    )

    # for converting to numpy
    input_tensor = input_tensor.detach()
    # (N, K)
    original_shape = input_tensor.shape
    # (K // group_size)
    num_lut = input_tensor.shape[1] // group_size

    # reshape to (N, K // group_size, group_size)
    input_tensor = input_tensor.reshape(input_tensor.shape[0], num_lut, group_size)
    from coremltools.models.neural_network.quantization_utils import (
        _get_kmeans_lookup_table_and_weight,
    )

    nbits = _DTYPE_TO_BIT_WIDTH[code_dtype]
    if nbits > 8:
        print(f"Requested nbits: {nbits}, rewriting to 8 bits to reduce the size")
        nbits = 8

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

    # reshape back to (N, K)
    res_w = res_w.reshape(*original_shape)

    return res_lut, res_w


@register_custom_op
def dequantize_codebook(
    codes: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Reconstructs the original tensor from codes and the codebook.

    Args:
        codes (torch.Tensor): Indices of codebook entries for each element
                              shape (N, K) for scalar quantization
        codebook (torch.Tensor): Codebook tensor used for quantization,
                                 shape (K // group_size, 2 ** nbits) where K is the dim 1 shape of input
        group_size (int): the group size for last dimension of Tensor, a slice of
        (`shape0`, `group_size`) elements will share the same lookup table
        output_dtype (torch.dtype): dtype for the output tensor.

    Returns:
        dequant (torch.Tensor): Reconstructed tensor, shape (N, K)

    """
    assert output_dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported output dtype: {output_dtype}"

    assert codes.shape[1] % group_size == 0
    K = codes.shape[1]
    num_lut = K // group_size
    # (N, K)
    original_shape = codes.shape

    # reshape to (N, num_lut, group_size)
    codes = codes.reshape(codes.shape[0], num_lut, group_size)
    dequant = torch.zeros_like(codes, dtype=output_dtype)

    # do lookup for each lookup table
    # dequant shape: (N, num_lut, group_size)
    # codebook shape: (num_lut, 2 ** nbits)
    # codes shape: (N, num_lut, group_size)
    for i in range(num_lut):
        # dequant[:, i, :]: (N, group_size)
        # using squeeze to remove the training dim 1s after the lookup
        dequant[:, i, :] = codebook[i][codes[:, i, :]].squeeze()

    dequant = dequant.reshape(*original_shape)
    return dequant.to(output_dtype)
