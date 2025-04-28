from typing import Tuple

import torch

import torchao.sparsity.marlin.utils as utils
from torchao.sparsity.marlin.utils import const
from torchao.sparsity.utils import mask_creator

__all__ = [
    "inject_24",
    "marlin_24_workspace",
    "pack_to_marlin_24",
    "unpack_from_marlin_24",
]


def inject_24(
    w: torch.Tensor, size_k: int, size_n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Injects 2:4 sparsity into a weight tensor. The sparsity is applied in a 2:4 ratio, where for every
    group of 4 weights, 2 will be pruned based on their value. The mask will be created based on the
    ranked weight values.

    Args:
        w (torch.Tensor): The weight tensor to inject sparsity into.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The pruned weight tensor and the mask tensor.
    """
    assert w.shape == (size_k, size_n)
    mask = mask_creator(w.t()).t().cuda().bool()
    return (mask * w).contiguous(), mask.contiguous()


def marlin_24_workspace(
    out_features: int,
    min_thread_n: int = const.MIN_THREAD_N,
    max_parallel: int = const.MAX_PARALLEL,
) -> torch.Tensor:
    """Creates a workspace for marlin 2:4 quantization. The workspace is used to coordinate the locks
    during the execution of the kernel.

    Args:
        out_features (int): The number of output features.
        min_thread_n (int, optional): The minimum number of threads per block. Defaults to `MARLIN_24_MIN_THREAD_N`.
        max_parallel (int, optional): The maximum number of parallel threads. Defaults to `MARLIN_24_MAX_PARALLEL`.
    Returns:
        torch.Tensor: The workspace tensor fully initialized with zeros.
    """
    assert out_features % min_thread_n == 0, (
        f"out_features = {out_features}, min_thread_n = {min_thread_n}"
    )
    max_workspace_size = (out_features // min_thread_n) * max_parallel
    return torch.zeros(max_workspace_size, dtype=torch.int, device="cuda")


def pack_to_marlin_24(
    q_w_24: torch.Tensor,
    scales: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs the quantized weights and scales into the marlin 2:4 format.

    Args:
        q_w_24 (torch.Tensor): The quantized weight tensor with 2:4 sparsity applied.
        scales (torch.Tensor): The scale tensor.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size that was applied during quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The packed quantized weights, the packed scales, and the meta tensor.
    """
    in_features, out_features = q_w_24.shape

    # Compress quantized weight
    q_w_24_comp, meta = _compress_quantized_24_weight(
        q_w_24, in_features, out_features, num_bits
    )

    in_features_comp = in_features // 2

    # Reformat to marlin
    marlin_24_q_w_comp = _to_marlin_weights(
        q_w_24_comp, in_features_comp, out_features, num_bits
    )

    marlin_24_s = _to_marlin_scales(
        scales, in_features, out_features, group_size, num_bits
    )

    return marlin_24_q_w_comp, marlin_24_s, meta


def unpack_from_marlin_24(
    q_w_24_comp: torch.Tensor,
    scales: torch.Tensor,
    meta: torch.Tensor,
    original_shape: torch.Size,
    group_size: int,
    num_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpacks the quantized weights and scales from the marlin 2:4 format.
    Args:
        q_w_24_comp (torch.Tensor): The packed quantized weights.
        scales (torch.Tensor): The packed scales.
        meta (torch.Tensor): The meta tensor.
        original_shape (torch.Size): The original shape of the weight tensor.
        group_size (int): The group size that was applied during quantization.
        num_bits (int): The number of bits used for quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The unpacked quantized weights and scales.
    """
    in_features, out_features = original_shape

    # Unpacks the scales
    unpacked_scales = _from_marlin_scale(scales, *original_shape, group_size, num_bits)

    in_features_comp = in_features // 2

    # Unpacks the weights
    unpacked_q_w_24_comp = _from_marlin_weights(
        q_w_24_comp, in_features_comp, out_features, num_bits
    )

    # Decompress quantized weight
    unpacked_q_w_24 = _decompress_quantized_24_weight(
        unpacked_q_w_24_comp, meta, in_features_comp, out_features, num_bits
    )

    return unpacked_q_w_24, unpacked_scales


def _compress_quantized_24_weight(
    q_24: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compresses the quantized weights to a 2:4 sparse format. Normalizes the weights over 0
    before compressing them.

    Args:
        q_24 (torch.Tensor): The quantized weight tensor.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The compressed quantized weight tensor and the meta tensor.
    """
    assert q_24.shape == (size_k, size_n)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp = q_24 - zp

    # Compress
    q_24_no_zp = q_24_no_zp.t().contiguous()
    q_24_no_zp_comp, meta = utils.sparse_semi_structured_from_dense_cutlass(q_24_no_zp)
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()

    # Restore zp
    q_24_comp = q_24_no_zp_comp + zp

    # Resize meta to its actual shape (without moving any data)
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    return q_24_comp, meta


def _decompress_quantized_24_weight(
    q_24_comp: torch.Tensor, meta: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Decompresses the quantized weights from a 2:4 sparse format and restores the original shape.

    Args:
        q_24_comp (torch.Tensor): The compressed quantized weight tensor in 2:4 sparse format.
        meta (torch.Tensor): The meta tensor.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
    Returns:
        torch.Tensor: The decompressed quantized weight tensor.
    """
    assert q_24_comp.shape == (size_k, size_n)

    # Resize meta back to its original shape
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp_comp = q_24_comp - zp

    # Decompress
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()
    q_24_no_zp = utils.sparse_semi_structured_to_dense_cutlass(q_24_no_zp_comp, meta)
    q_24_no_zp = q_24_no_zp.t().contiguous()

    # Revert meta resize
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    # Restore zp
    q_24 = q_24_no_zp + zp

    return q_24


def _to_marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Converts a quantized and 2:4 sparse format weight tensor to the marlin 2:4 format.

    Args:
        q_w (torch.Tensor): The quantized weight tensor in 2:4 sparse format.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
    Returns:
        torch.Tensor: The weight tensor in the marlin 2:4 format.
    """
    # Permute
    perm_24, _, _ = utils.get_perms_24(num_bits)
    q_w = utils.marlin_permute_weights(q_w, size_k, size_n, perm_24)

    # Pack
    pack_factor = utils.get_pack_factor(num_bits)
    orig_device = q_w.device

    # Original implementation uses numpy + uint32 but we need to use int64 because torch.uint32
    # does not support rshift_cpu.
    q_w = q_w.cpu().to(torch.int64)
    q_packed = torch.zeros(
        (q_w.shape[0], q_w.shape[1] // pack_factor),
        dtype=torch.int64,
        device=q_w.device,
    )
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << (num_bits * i)

    q_packed = q_packed.to(orig_device, dtype=torch.int32)
    return q_packed


def _from_marlin_weights(
    q_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Converts a weight tensor in the marlin 2:4 format to a regular quantized 2:4 sparse format.

    Args:
        q_packed (torch.Tensor): The weight tensor in the marlin 2:4 format.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
    Returns:
        torch.Tensor: The weight tensor in the quantized 2:4 sparse format.
    """
    perm_24, _, _ = utils.get_reverse_perms_24(num_bits)

    pack_factor = utils.get_pack_factor(num_bits)
    orig_device = q_packed.device

    # Unpack from marlin format.
    # Original implementation uses numpy + uint32 but we need to use int64 because torch.uint32
    # does not support rshift_cpu.
    q_packed = q_packed.cpu().to(torch.int64)
    q_w_unpacked = torch.zeros(
        (q_packed.shape[0], q_packed.shape[1] * pack_factor),
        dtype=torch.int64,
        device=q_packed.device,
    )
    for i in range(pack_factor):
        q_w_unpacked[:, i::pack_factor] = (q_packed >> (num_bits * i)) & (
            (1 << num_bits) - 1
        )

    q_w_unpacked = q_w_unpacked.to(orig_device, dtype=torch.int32)

    q_w_comp = utils.reverse_marlin_permute_weights(
        q_w_unpacked, size_k, size_n, perm_24
    )
    return q_w_comp


def _to_marlin_scales(
    scales: torch.Tensor, size_k: int, size_n: int, group_size: int, num_bits: int
) -> torch.Tensor:
    """Converts a scale tensor to the format necessary for marlin.
    Args:
        scales (torch.Tensor): The scale tensor.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        group_size (int): The group size that was applied during quantization.
        num_bits (int): The number of bits used for quantization.

    Returns:
        torch.Tensor: The scale tensor in the marlin format.
    """
    _, scale_perm_24, scale_perm_single_24 = utils.get_perms_24(num_bits)
    if group_size < size_k and group_size != -1:
        scales = scales.reshape((-1, len(scale_perm_24)))[:, scale_perm_24]
    else:
        scales = scales.reshape((-1, len(scale_perm_single_24)))[
            :, scale_perm_single_24
        ]
    scales = scales.reshape((-1, size_n)).contiguous()
    return scales


def _from_marlin_scale(
    scales: torch.Tensor, size_k: int, size_n: int, group_size: int, num_bits: int
) -> torch.Tensor:
    """Converts a scale tensor from the marlin format to their original format.

    Args:
        scales (torch.Tensor): The scale tensor in the marlin format.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        group_size (int): The group size that was applied during quantization.
        num_bits (int): The number of bits used for quantization.
    Returns:
        torch.Tensor: The scale tensor in their original format
    """
    _, scale_perm_24, scale_perm_single_24 = utils.get_reverse_perms_24(num_bits)
    if group_size < size_k and group_size != -1:
        scales = scales.reshape((-1, len(scale_perm_24)))[:, scale_perm_24]
        return scales.reshape((size_k // group_size, size_n))
    else:
        scales = scales.reshape((-1, len(scale_perm_single_24)))[
            :, scale_perm_single_24
        ]
        return scales.reshape((1, -1))
