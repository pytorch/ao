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
        input (torch.Tensor): Input tensor to quantize, shape (out_features, in_features).
        codebook (torch.Tensor): Codebook tensor for quantization, shape (k, out_block_size, in_block_size).
        chunk_size (int): Number of elements to process per chunk to control memory usage.
        code_dtype (torch.dtype): dtype for the codes.

    Output:
        codes (torch.Tensor): indices of the closest codebook entries for each block.
    """
    if code_dtype in _SUB_BYTE_UINT_BOUNDS:
        code_dtype = torch.uint8
    assert (
        input.dim() == 2
    ), (
        f"expect input tensor dim == 2 but got dim = {input.dim()}"
    )  # not sure if I should do this
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"
    assert (
        codebook.dim() == input.dim() + 1
    ), f"codebook dim ({codebook.dim()}) must be input dim + 1 ({input.dim() + 1})"

    k, out_block_size, in_block_size = codebook.shape
    out_features, in_features = input.shape
    assert (
        out_features % out_block_size == 0
    ), f"out_features ({out_features}) must be divisible by out_block_size ({out_block_size})."
    assert (
        in_features % in_block_size == 0
    ), f"in_features ({in_features}) must be divisible by in_block_size ({in_block_size})."

    num_out_blocks = out_features // out_block_size
    num_in_blocks = in_features // in_block_size

    num_scale_blocks = scales.shape[1]

    input = input.view(out_features, num_scale_blocks, -1)
    input = input / scales

    input = input.view(num_out_blocks, out_block_size, num_in_blocks, in_block_size)
    input_flat = input.permute(0, 2, 1, 3).reshape(
        num_out_blocks * num_in_blocks, out_block_size * in_block_size
    )

    codebook_flat = codebook.reshape(k, out_block_size * in_block_size)

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

    codes = codes.view(num_out_blocks, num_in_blocks)

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
                                          shape (num_out_blocks, num_in_blocks).
        codebook (torch.Tensor): Codebook tensor used for quantization,
                                 shape (k, out_block_size, in_block_size).
        output_dtype (torch.dtype): dtype for the output tensor.

    Returns:
        dequant (torch.Tensor): Reconstructed tensor, shape (out_features, in_features), where
                      out_features = num_out_blocks * out_block_size and in_features = num_in_blocks * in_block_size.
    """
    assert output_dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported output dtype: {output_dtype}"

    _, out_block_size, in_block_size = codebook.shape
    num_out_blocks, num_in_blocks = codes.shape

    # Use codes to lookup corresponding codebook entries and reshape
    dequant = codebook[
        codes
    ]  # shape: [num_out_blocks, num_in_blocks, out_block_size, in_block_size]

    dequant = dequant.permute(0, 2, 1, 3).reshape(
        num_out_blocks * out_block_size, num_in_blocks * in_block_size
    )

    dequant = dequant.view(num_out_blocks * out_block_size, scales.shape[1], -1)
    dequant = dequant * scales

    dequant = dequant.view(num_out_blocks * out_block_size, -1)

    return dequant.to(output_dtype)


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
        code_dtype (torch.dtype): The dtype for the codes.
        max_iter (int): Maximum number of k-means iterations.
        devices (List[torch.device]): Devices to run k-means on.

    Returns:
        torch.Tensor: The codebook tensor, shape (codebook_size, block_height, block_width).
    """
    if code_dtype == torch.int32:
        codebook_size = 2**16
    else:
        codebook_size = _DTYPE_TO_QVALUE_BOUNDS[code_dtype][1] + 1

    out_block_size, in_block_size = block_size
    out_features, in_features = input_tensor.shape
    num_out_blocks = out_features // out_block_size
    num_in_blocks = in_features // in_block_size

    assert (
        in_features % scale_block_size == 0
    ), f"input_features ({in_features}) must be divisible by scale_block_size ({scale_block_size})."
    num_scale_blocks = in_features // scale_block_size

    input = input_tensor.view(out_features, num_scale_blocks, scale_block_size)

    # Not sure if I should make scales max only when block_size is (1, 1)
    if block_size == (1, 1):
        scales = input.max(
            dim=(-1), keepdim=True
        ).values  # Shape: [out_features, num_scale_blocks, 1]
    else:
        scales = input.norm(
            dim=(-1), keepdim=True
        )  # Shape: [out_features, num_scale_blocks, 1]
    scales = torch.clamp(scales, min=1e-9)

    input = input / scales

    input = input.view(num_out_blocks, out_block_size, num_in_blocks, in_block_size)

    input = input.permute(0, 2, 1, 3).reshape(
        num_out_blocks * num_in_blocks, out_block_size * in_block_size
    )

    codebook, _, _ = fit_kmeans(
        input,
        k=codebook_size,
        max_iter=max_iter,
        devices=devices,
    )

    return codebook.view(codebook_size, out_block_size, in_block_size), scales


def fit_kmeans(
    data: torch.Tensor,
    k: int,
    max_iter: int = 200,
    check_every: int = 10,
    rtol: float = 1e-06,
    atol: float = 1e-08,
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
