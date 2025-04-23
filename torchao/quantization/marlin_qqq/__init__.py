from typing import Tuple

import torch

from torchao.quantization.granularity import (
    PerAxis,
    PerGroup,
)
from torchao.quantization.marlin_qqq.utils import (
    const,
    get_pack_factor,
    get_qqq_scale_perms,
    get_qqq_scale_reverse_perms,
    get_qqq_weight_perm,
    get_qqq_weight_reverse_perm,
    marlin_permute_weights,
    reverse_marlin_permute_weights,
)

__all__ = [
    "marlin_qqq_workspace",
    "pack_to_marlin_qqq",
    "unpack_from_marlin_qqq",
]


def marlin_qqq_workspace(
    out_features: int,
    min_thread_n: int = const.MIN_THREAD_N,
    max_parallel: int = const.MAX_PARALLEL,
) -> torch.Tensor:
    """Creates a workspace for marlin qqq. The workspace is used to coordinate the locks
    during the execution of the kernel.

    Args:
        out_features (int): The number of output features.
        min_thread_n (int, optional): The minimum number of threads per block. Defaults to `MARLIN_QQQ_MIN_THREAD_N`.
        max_parallel (int, optional): The maximum number of parallel threads. Defaults to `MARLIN_QQQ_MAX_PARALLEL`.
    Returns:
        torch.Tensor: The workspace tensor fully initialized with zeros.
    """
    assert out_features % min_thread_n == 0, (
        f"out_features = {out_features}, min_thread_n = {min_thread_n}"
    )
    max_workspace_size = (out_features // min_thread_n) * max_parallel
    return torch.zeros(max_workspace_size, dtype=torch.int, device="cuda")


def pack_to_marlin_qqq(
    q_w: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack the quantized weights and scales to the marlin format.

    Args:
        q_w (torch.Tensor): The quantized weight.
        s_group (torch.Tensor): The per-group quantization scale.
        s_channel (torch.Tensor): The per-channel quantization scale.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The packed quantized weight in marlin format, the packed per-group scale in marlin format, and the packed per-channel scale in marlin format.
    """
    in_features, out_features = q_w.shape

    assert num_bits == 4, "Marlin QQQ only supports 4-bit for now."

    # Reformat to marlin
    marlin_qqq_q_w = _to_marlin_weights(
        q_w, in_features, out_features, num_bits, group_size
    )
    marlin_qqq_s_group, marlin_qqq_s_channel = _to_marlin_scales(
        s_group, s_channel, in_features, out_features, num_bits, group_size
    )

    return marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel


def _to_marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    """Converts a quantized weight tensor to the marlin format.

    Args:
        q_w (torch.Tensor): The quantized weight.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size of quantization.
    Returns:
        torch.Tensor: The packed quantized weight in marlin format.
    """
    if group_size == -1:
        group_size = size_k
    granularity = PerAxis(1) if group_size == size_k else PerGroup(group_size)
    # Permute
    perm = get_qqq_weight_perm(num_bits, granularity)
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    # q_w is torch.uint32 originally, but torch does not support lshift_cuda or lshift_cpu, we have to
    # convert it to torch.int64
    q_w = q_w.to(torch.int64)
    q_packed = torch.zeros(
        (q_w.shape[0], q_w.shape[1] // pack_factor),
        dtype=torch.int64,
        device=q_w.device,
    )

    if group_size == size_k:
        for i in range(pack_factor):
            q_packed |= (q_w[:, i::pack_factor] & 0xF) << (num_bits * i)
    else:
        for i in range(pack_factor):
            q_packed |= q_w[:, i::pack_factor] << (num_bits * i)

    q_packed = q_packed.to(torch.int32).to(orig_device)
    return q_packed


def _to_marlin_scales(
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts the per-group scale and the per-channel scale to the format necessary for marlin.

    Args:
        s_group (torch.Tensor): The per-group quantization scale.
        s_channel (torch.Tensor): The per-channel quantization scale.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scale tensors in the marlin format.
    """
    if group_size == -1:
        group_size = size_k
    scale_perm, scale_perm_single = get_qqq_scale_perms(num_bits)
    if group_size < size_k:
        s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[
            :, scale_perm_single
        ]
        s_group = s_group.reshape((-1, size_n)).contiguous()
    else:
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[
            :, scale_perm_single
        ]
    s_channel = s_channel.reshape((-1, size_n)).contiguous()

    return s_group, s_channel


def unpack_from_marlin_qqq(
    q_w: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    original_shape: torch.Size,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpacks the quantized weights and scales from the marlin format.
    Args:
        q_w (torch.Tensor): The packed quantized weights.
        s_group (torch.Tensor): The per-group quantization scale.
        s_channel (torch.Tensor): The per-channel quantization scale.
        original_shape (torch.Size): The original shape of the weight tensor.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The unpacked quantized weights and scales.
    """
    in_features, out_features = original_shape

    assert num_bits == 4, "Marlin QQQ only supports 4-bit for now."

    # Unpacks the scales
    unpacked_s_group, unpacked_s_channel = _from_marlin_scales(
        s_group, s_channel, in_features, out_features, num_bits, group_size
    )

    # Unpacks the weights
    unpacked_q_w = _from_marlin_weights(
        q_w, in_features, out_features, num_bits, group_size
    )

    return unpacked_q_w, unpacked_s_group, unpacked_s_channel


def _from_marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    """Converts a weight tensor in the marlin format to a regular format.
    Args:
        q_w (torch.Tensor): The packed quantized weights.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size of quantization.
    Returns:
        torch.Tensor: The unpacked quantized weights.
    """
    if group_size == -1:
        group_size = size_k
    granularity = PerAxis(1) if group_size == size_k else PerGroup(group_size)
    # Permute
    perm = get_qqq_weight_reverse_perm(num_bits, granularity)

    orig_device = q_w.device

    wf = (
        torch.tensor(list(range(0, 32, num_bits)), dtype=torch.int32)
        .unsqueeze(0)
        .to(orig_device)
    )
    # unpack weight
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(q_w, 2).expand(-1, -1, 32 // num_bits),
        wf.unsqueeze(0),
    )
    weight = torch.bitwise_and(weight, (2**num_bits) - 1)
    weight = weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])
    q_w_comp = reverse_marlin_permute_weights(weight, size_k, size_n, perm)

    return q_w_comp


def _from_marlin_scales(
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts the quantization scales from the marlin format to their original format.
    Args:
        s_group (torch.Tensor): The per-group quantization scale in marlin format.
        s_channel (torch.Tensor): The per-channel quantization scale in marlin format.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The per-group quantization scale in the original format and
        the per-channel quantization scale in the original format.
    """
    if group_size == -1:
        group_size = size_k
    scale_perm, scale_perm_single = get_qqq_scale_reverse_perms(num_bits)
    if group_size < size_k:
        s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[
            :, scale_perm_single
        ]
        s_group = s_group.reshape((-1, size_n)).contiguous()
    else:
        s_channel = s_channel.reshape((-1, len(scale_perm_single)))[
            :, scale_perm_single
        ]
    s_channel = s_channel.reshape((-1, size_n)).contiguous()

    return s_group, s_channel
