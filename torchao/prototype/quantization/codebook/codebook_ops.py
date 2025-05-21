# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional, Tuple

import torch

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_QVALUE_BOUNDS,
    _SUB_BYTE_UINT_BOUNDS,
)


def quantize_codebook(
    input: torch.Tensor,
    codebook: torch.Tensor,
    scales: torch.Tensor,
    chunk_size: int = 1024,
    code_dtype: torch.dtype = torch.uint4,
) -> torch.Tensor:
    """
    code modified from: https://github.com/Vahe1994/AQLM/blob/main/src/kmeans.py

    Args:
        input (torch.Tensor): Input tensor to quantize, shape (d1, d2, ..., dN).
        codebook (torch.Tensor): Codebook tensor for quantization, shape (k, b1, b2, ..., bN) where b_i are block sizes.
        scales (torch.Tensor): Scales, shape (d1, d2, ..., dN // scale_block_size, 1).
        chunk_size (int): Number of elements to process per chunk to control memory usage.
        code_dtype (torch.dtype): dtype for the codes.

    Output:
        codes (torch.Tensor): indices of the closest codebook entries for each block, shape (d1//b1, d2//b2, ..., dN//bN).
    """
    if code_dtype in _SUB_BYTE_UINT_BOUNDS:
        code_dtype = torch.uint8
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"
    assert codebook.dim() == input.dim() + 1, (
        f"codebook dim ({codebook.dim()}) must be input dim + 1 ({input.dim() + 1})"
    )

    k = codebook.shape[0]
    block_size = codebook.shape[1:]
    input_size = input.shape
    D = input.dim()
    for i in range(D):
        assert (input_size[i] % block_size[i]) == 0, (
            f"dimension {i} of input ({input_size[i]}) must be divisible by block_size ({block_size[i]})."
        )

    num_scale_blocks = scales.shape[-2]

    new_shape = input_size[:-1] + (num_scale_blocks, -1)
    input_reshaped = input.view(
        *new_shape
    )  # shape: (d1, d2, ..., num_scale_blocks, scale_block_size)
    input_reshaped = input_reshaped / scales

    input = input_reshaped.view(*input_size)
    input_flat = _reshape_into_blocks(
        input, block_size
    )  # shape: (num_blocks, block_vector_size)

    codebook_flat = codebook.reshape(k, -1)

    codes = torch.empty(input_flat.size(0), dtype=torch.int64, device=input.device)

    # Process in chunks to avoid memory spikes
    for chunk_start in range(0, input_flat.size(0), chunk_size):
        chunk_end = min(chunk_start + chunk_size, input_flat.size(0))

        input_chunk = input_flat[chunk_start:chunk_end]
        input_chunk = input_chunk

        # Compute distances and find nearest codebook entries for the chunk
        distances = torch.addmm(
            torch.bmm(codebook_flat[:, None, :], codebook_flat[:, :, None]).flatten(),
            input_chunk,
            codebook_flat.T,
            beta=-0.5,
        )
        # distance = || input_chunk[:, None, :] - codebook_flat[None, :, :] ||^2 = || input_chunk ||^2 + || codebook_flat ||^2 - 2 * input_chunk @ codebook_flat.T
        # don't need to compute input_chunk squared norm as it's constant during argmax

        codes[chunk_start:chunk_end] = distances.argmax(dim=1)

    block_grid_shape = [input_size[i] // block_size[i] for i in range(D)]
    codes = codes.view(*block_grid_shape)  # shape: (d1//b1, d2//b2, ..., dN//bN)

    return codes.to(code_dtype)


def dequantize_codebook(
    codes: torch.Tensor,
    codebook: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Reconstructs the original tensor from codes and the codebook.

    Args:
        codes (torch.Tensor): Indices of codebook entries for each block,
                                          shape (d1//b1, d2//b2, ..., dN//bN).
        codebook (torch.Tensor): Codebook tensor used for quantization,
                                 shape (k, b1, b2, ..., bN) where b_i are block sizes.
        scales (torch.Tensor): Scales, shape (d1, d2, ..., dN // scale_block_size, 1).
        output_dtype (torch.dtype): dtype for the output tensor.

    Returns:
        dequant (torch.Tensor): Reconstructed tensor, shape (out_features, in_features)
    """
    assert output_dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported output dtype: {output_dtype}"

    block_size = codebook.shape[1:]
    block_grid_shape = codes.shape
    D = codebook.dim() - 1
    original_shape = [block_grid_shape[i] * block_size[i] for i in range(D)]

    # Use codes to lookup corresponding codebook entries and reshape
    dequant = codebook[codes]  # shape: (*block_grid_shape, *block_size)

    # probably can make this simpler
    dequant = _reshape_from_blocks(
        dequant.view(-1, int(torch.prod(torch.tensor(block_size)))),
        block_size,
        tuple(original_shape),
    )

    num_scale_blocks = scales.shape[-2]

    new_shape = dequant.shape[:-1] + (num_scale_blocks, -1)
    dequant = dequant.view(
        *new_shape
    )  # (d1, d2, ..., num_scale_blocks, scale_block_size)
    dequant.mul_(scales)

    dequant = dequant.view(*original_shape)

    return dequant.to(output_dtype)


@torch.no_grad()
def choose_qparams_codebook(
    input_tensor: torch.Tensor,
    block_size: Tuple[int, ...],
    scale_block_size: int,
    code_dtype: torch.dtype,
    max_iter: int = 200,
    devices: Optional[List[torch.device]] = None,
) -> torch.Tensor:
    """
    Initialize the codebook using k-means clustering on blocks of the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to be quantized.
        block_size (Tuple[int, ...],): The size of the blocks for k-means clustering.
        scale_block_size (int): The size of the blocks that share a scale
        code_dtype (torch.dtype): The dtype for the codes.
        max_iter (int): Maximum number of k-means iterations.
        devices (List[torch.device]): Devices to run k-means on.

    Returns:
        torch.Tensor: The codebook tensor, shape (codebook_size, *block_size).
    """
    if code_dtype == torch.int32:
        codebook_size = 2**16
    else:
        codebook_size = _DTYPE_TO_QVALUE_BOUNDS[code_dtype][1] + 1

    input_size = input_tensor.shape
    D = input_tensor.dim()
    for i in range(D):
        assert input_size[i] % block_size[i] == 0, (
            f"Dimension {i} must be divisible by block_size {block_size[i]}."
        )

    assert input_tensor.shape[-1] % scale_block_size == 0, (
        f"input_features ({input_tensor.shape[-1]}) must be divisible by scale_block_size ({scale_block_size})."
    )
    num_scale_blocks = input_tensor.shape[-1] // scale_block_size

    new_shape = list(input_size[:-1]) + [num_scale_blocks, scale_block_size]
    input = input_tensor.view(*new_shape)

    # Not sure if I should make scales max only when block_size is (1, 1)
    if block_size == (1, 1):
        scales = input.max(
            dim=(-1), keepdim=True
        ).values  # Shape: [*input_size[:-1], num_scale_blocks, 1]
    else:
        scales = input.norm(
            dim=(-1), keepdim=True
        )  # Shape: [*input_size[:-1], num_scale_blocks, 1]
    scales = torch.clamp(scales, min=1e-9)

    input = input / scales

    input = input.view(*input_size)

    input = _reshape_into_blocks(
        input, block_size
    )  # Shape: (num_blocks, block_vector_size)

    codebook, _, _ = fit_kmeans(
        input,
        k=codebook_size,
        max_iter=max_iter,
        devices=devices,
    )

    return codebook.view(codebook_size, *block_size), scales


@torch.jit.script
def _kmeans_greedy_init(data: torch.Tensor, k: int) -> torch.Tensor:
    # code modified from: https://github.com/Vahe1994/AQLM/blob/main/src/kmeans.py not sure if I should modify for
    clusters = torch.zeros(k, data.shape[1], device=data.device)
    running_min_distances = torch.full(
        (data.shape[0],), torch.inf, device=data.device, dtype=data.dtype
    )
    data_norm_squared = data.norm(p=2, dim=1).square()

    for i in range(k):
        clusters[i] = data[running_min_distances.argmax()]
        distances_to_cluster_i = (
            data_norm_squared - 2 * data @ clusters[i] + clusters[i].norm().square()
        )
        running_min_distances = torch.minimum(
            running_min_distances, distances_to_cluster_i, out=running_min_distances
        )
    return clusters


@torch.jit.script
def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 200,
    check_every: int = 10,
    rtol: float = 1e-06,
    atol: float = 1e-08,
    greedy_init: bool = True,
    block_size_vals: int = 2**30,
    devices: Optional[List[torch.device]] = None,
):
    """
    code modified from: https://github.com/Vahe1994/AQLM/blob/main/src/kmeans.py not sure if I should modify for
    :param data: [nsamples, dim]
    :param k: number of centroids
    :param max_iter: run at most this many iterations
    :param check_every: check for convergence (allclose(new_centroids, old_centroids)) once in this many steps
    :param rtol: early stopping relative tolerance for centroids
    :param atol: early stopping absolute tolerance for centroids
    :param block_size_vals: how many dot products to compute at a time
    :param devices: if specified, run kmeans in data-parallel mode across these devices
    :return: (clusters float[k, dim], data_indices int[nsamples], reconstructed_data: float[nsamples, dim])
    """
    if devices is None:
        devices = [data.device]

    if greedy_init:
        clusters = _kmeans_greedy_init(data, k)
    else:
        clusters = data[torch.randperm(data.shape[0])[:k], :]  # [k, dim]

    block_size = block_size_vals // k
    shard_size = (len(data) - 1) // len(devices) + 1
    data = [
        data[gi * shard_size : (gi + 1) * shard_size].to(devices[gi], non_blocking=True)
        for gi in range(len(devices))
    ]
    nearest_indices = [
        torch.empty(len(data[gi]), dtype=torch.int64, device=devices[gi])
        for gi in range(len(devices))
    ]
    clusters = [clusters.to(device, non_blocking=True) for device in devices]

    for i in range(max_iter):
        for block_start in range(0, shard_size, block_size):
            for gi in range(len(devices)):
                nearest_indices[gi][block_start : block_start + block_size] = (
                    torch.addmm(
                        torch.bmm(
                            clusters[gi][:, None, :], clusters[gi][:, :, None]
                        ).flatten(),
                        data[gi][block_start : block_start + block_size],
                        clusters[gi].T,
                        beta=-0.5,
                    ).argmax(1)
                )
            # note: the above formula equals to - 0.5 || data[:, None, :] - clusters[None, :, :] || ^ 2 + const

        if len(devices) == 1:
            new_clusters = [
                clusters[0]
                .clone()
                .index_reduce_(
                    dim=0,
                    index=nearest_indices[0],
                    source=data[0],
                    reduce="mean",
                    include_self=False,
                )
            ]
        else:
            cluster_sums = [
                torch.zeros_like(clusters[gi])
                .index_add(dim=0, index=nearest_indices[gi], source=data[gi])
                .to(devices[0], non_blocking=True)
                for gi in range(len(devices))
            ]
            cluster_counts = [
                torch.bincount(nearest_indices[gi], minlength=k).to(
                    devices[0], non_blocking=True
                )
                for gi in range(len(devices))
            ]
            for gi in range(1, len(devices)):
                cluster_sums[0] += cluster_sums[gi]
                cluster_counts[0] += cluster_counts[gi]

            new_clusters = [
                cluster_sums[0] / cluster_counts[0].unsqueeze(1).clamp_min(1)
            ]
            new_clusters[0] += (cluster_counts[0].unsqueeze(1) == 0) * clusters[0]
            for gi in range(1, len(devices)):
                new_clusters.append(new_clusters[0].to(devices[gi], non_blocking=True))

        if i % check_every == 0:
            if torch.allclose(new_clusters[0], clusters[0], rtol=rtol, atol=atol):
                break
        clusters = new_clusters
    for block_start in range(0, shard_size, block_size):
        for gi in range(len(devices)):
            nearest_indices[gi][block_start : block_start + block_size] = torch.addmm(
                torch.bmm(clusters[gi][:, None, :], clusters[gi][:, :, None]).flatten(),
                data[gi][block_start : block_start + block_size],
                clusters[gi].T,
                beta=-0.5,
            ).argmax(1)

    clusters = clusters[0]
    nearest_indices = torch.cat(
        [nearest_indices[gi].to(devices[0]) for gi in range(len(devices))], dim=0
    )
    reconstructed_data = clusters[nearest_indices]
    return clusters, nearest_indices, reconstructed_data


def _reshape_into_blocks(
    input: torch.Tensor, block_size: Tuple[int, ...]
) -> torch.Tensor:
    """
    Reshape an N-D input tensor into a 2D tensor where each row corresponds to one block.
    """
    assert len(block_size) == input.dim(), (
        f"block_size {block_size} must match the input dimension {input.dim()}"
    )
    input_size = input.shape

    # Create shape with alternating (num_blocks_along_dim, block_size_dim)
    reshaped_dims = []
    for i in range(input.dim()):
        assert input_size[i] % block_size[i] == 0, (
            f"Input size at dim {i} ({input_size[i]}) must be divisible by block_size[i] ({block_size[i]})."
        )
        reshaped_dims.extend([input_size[i] // block_size[i], block_size[i]])

    input_reshaped = input.view(*reshaped_dims)  # Shape: [g1, b1, g2, b2, ..., gD, bD]

    D = input.dim()
    perm_order = list(range(2 * D))
    grid_dims = perm_order[0::2]
    block_dims = perm_order[1::2]
    perm_order = grid_dims + block_dims

    input_reshaped = input_reshaped.permute(
        *perm_order
    )  # Shape: [g1, g2, ..., gD, b1, b2, ..., bD]

    num_blocks = 1
    for i in range(D):
        num_blocks *= input_size[i] // block_size[i]
    block_vector_size = 1
    for b in block_size:
        block_vector_size *= b

    input_flat = input_reshaped.reshape(num_blocks, block_vector_size)
    return input_flat


def _reshape_from_blocks(
    blocks: torch.Tensor, block_size: Tuple[int, ...], original_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Reshape from the 2D block form (num_blocks, block_vector_size) back to the original N-D shape.
    """
    D = len(block_size)

    reshaped_dims = []
    num_blocks = 1
    for i in range(D):
        reshaped_dims.extend([original_shape[i] // block_size[i], block_size[i]])
        num_blocks *= original_shape[i] // block_size[i]
    block_vector_size = 1
    for b in block_size:
        block_vector_size *= b

    perm_order = []
    for i in range(D):
        perm_order.append(i * 2)  # grid dim indices first
    for i in range(D):
        perm_order.append(i * 2 + 1)  # block dim indices after

    perm_inverse = [0] * (2 * D)
    for idx, val in enumerate(perm_order):
        perm_inverse[val] = idx

    input_permuted_shape = []
    for i in range(D):
        input_permuted_shape.append(original_shape[i] // block_size[i])
    for i in range(D):
        input_permuted_shape.append(block_size[i])

    blocks_permuted = blocks.view(
        *input_permuted_shape
    )  # Shape: [g1, g2, ..., gD, b1, b2, ..., bD]

    blocks_unpermuted = blocks_permuted.permute(
        *perm_inverse
    )  # Shape: [g1, b1, g2, b2, ..., gD, bD]

    return blocks_unpermuted.reshape(*original_shape)
