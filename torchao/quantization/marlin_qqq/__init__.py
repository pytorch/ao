from typing import List, Tuple

import numpy
import torch

from torchao.quantization.marlin_qqq.utils import (
    marlin_permute_weights,
    get_pack_factor,
    qqq_quantize_weights,
    const,
)

__all__ = [
    "marlin_qqq_workspace",
    "marlin_qqq_weights",
    "get_qqq_scale_perms",
    "get_qqq_weight_perm",
    "marlin_qqq_permute_scales",
    "marlin_qqq_quantize",
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
    assert (
        out_features % min_thread_n == 0
    ), f"out_features = {out_features}, min_thread_n = {min_thread_n}"
    max_workspace_size = (out_features // min_thread_n) * max_parallel
    return torch.zeros(max_workspace_size, dtype=torch.int, device="cuda")


def marlin_qqq_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    perm: torch.Tensor,
    group_size: int,
):
    """Pack the quantized weights to the marlin format.

    Args:
        q_w (torch.Tensor): The quantized weight.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        num_bits (int): The number of bits used for quantization.
        perm (torch.Tensor): The weight permutation tensor.
        group_size (int): The group size of quantization.
    Returns:
        torch.Tensor: The packed weight in marlin format.
    """
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_packed = numpy.zeros(
        (q_w.shape[0], q_w.shape[1] // pack_factor), dtype=numpy.uint32
    )
    if group_size == size_k:
        for i in range(pack_factor):
            q_packed |= (q_w[:, i::pack_factor] & 0xF) << num_bits * i
    else:
        for i in range(pack_factor):
            q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(numpy.int32)).to(orig_device)

    return q_packed


def get_qqq_scale_perms() -> Tuple[List[int], List[int]]:
    """Precompute permutations for the marlin scale shuffling.

    Returns:
        Tuple[List[int], List[int]]: Scale permutation list and
        scale permutation list for a single group.
    """

    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


# NOTE(HandH1998): QQQ employs different perms for per-group and per-channel weight quantization. # noqa: E501
def get_qqq_weight_perm(num_bits: int, quant_type: str) -> torch.Tensor:
    """Precompute permutations for the marlin weight shuffling.

    Args:
        num_bits (int): Number of bits to pack.
        quant_type (str): The weight quantization type: per-group or per-channel.
    Returns:
        torch.Tensor: The weight permutation tensor.
    """
    perm_list: List[int] = []
    for i in range(32):
        perm1: List[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                4 * (i % 4),
                4 * (i % 4) + 1,
                4 * (i % 4) + 2,
                4 * (i % 4) + 3,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = numpy.array(perm_list)

    assert quant_type in ["per-channel", "per-group"], "not supported quantization type"
    if num_bits == 4:
        if quant_type == "per-channel":
            interleave = numpy.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    else:
        raise Exception("num_bits must be 4, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def marlin_qqq_permute_scales(
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts per-group scale and per-channel scale to the format necessary for marlin.

    Args:
        s_group (torch.Tensor): The per-group quantization scale.
        s_channel (torch.Tensor): The per-channel quantization scale.
        size_k (int): The number of input features.
        size_n (int): The number of output features.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scale tensors in the marlin format.
    """
    scale_perm, scale_perm_single = get_qqq_scale_perms()
    if group_size < size_k and group_size != -1:
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


def marlin_qqq_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the weight and convert it to the marlin format.

    Args:
        w (torch.Tensor): The original weight.
        num_bits (int): The number of bits for quantization.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        The dequantized weight, the quantized weight in marlin format, the per-group quantization scale in marlin format,
        and the per-channel quantization scale in marlin format.
    """
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k
    quant_type = "per-channel" if group_size == size_k else "per-group"

    # Quantize
    w_ref, q_w, s_group, s_channel = qqq_quantize_weights(w, num_bits, group_size)

    # Reformat to marlin_qqq
    weight_perm = get_qqq_weight_perm(num_bits, quant_type)
    marlin_qqq_q_w = marlin_qqq_weights(
        q_w, size_k, size_n, num_bits, weight_perm, group_size
    )
    marlin_qqq_s_group, marlin_qqq_s_channel = marlin_qqq_permute_scales(
        s_group, s_channel, size_k, size_n, group_size
    )

    # Create result
    res_list = [w_ref, marlin_qqq_q_w, marlin_qqq_s_group, marlin_qqq_s_channel]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list
