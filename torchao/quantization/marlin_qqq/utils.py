import torch
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarlinQQQConstants:
    TILE: int = 16
    MIN_THREAD_N: int = 64
    MAX_PARALLEL: int = 16

    SUPPORTED_NUM_BITS: List[int] = field(default_factory=lambda: [4])
    SUPPORTED_GROUP_SIZES: List[int] = field(default_factory=lambda: [-1, 128])


const = MarlinQQQConstants()


def get_pack_factor(num_bits: int) -> int:
    """Compute the packing factor for a given number of bits.

    Args:
        num_bits (int): Number of bits to pack.
    Returns:
        int: The packing factor.
    """
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def marlin_permute_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    perm: torch.Tensor,
    tile: int = const.TILE,
) -> torch.Tensor:
    """Permute weights to 16x64 Marlin tiles.

    Args:
        q_w (torch.Tensor): Quantized weights.
        size_k (int): Number of input features.
        size_n (int): Number of output features.
        perm (torch.Tensor): The computed permutation tensor to be applied.
        tile (int, optional): Tile size. Defaults to `TILE`.
    Returns:
        torch.Tensor: Weight tensor permuted to Marlin tiles.
    """
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


# QQQ employs different quant schemes for per-group and
# per-channel quantization.
def qqq_quantize_weights(
    w: torch.Tensor, num_bits: int, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the weight.

    Args:
        w (torch.Tensor): The original weight.
        num_bits (int): The number of bits for quantization.
        group_size (int): The group size of quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        The dequantized weight, the quantized weight, the per-group quantization scale,
        and the per-channel quantization scale.
    """
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"
    assert num_bits in const.SUPPORTED_NUM_BITS, f"Unsupported num_bits = {num_bits}"
    assert group_size in const.SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if group_size < size_k:
        # Reshape to [groupsize, -1]
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        # Compute scale for each group
        s_group = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_group *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        q_w = torch.round(w / s_group).int()
        q_w += half_q_val
        q_w = torch.clamp(q_w, 0, max_q_val)
        # Compute ref (dequantized)
        w_ref = (q_w - half_q_val).half() * s_group

        # Restore original shapes
        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

        # Compute int8 quantization scale for each channel
        s_channel = torch.max(torch.abs(w_ref), 0, keepdim=True)[0]
        s_channel /= 127.0
        t_int8 = (w_ref / s_channel).round().clamp(-128, 127).to(torch.int8)
        w_ref = t_int8.half() * s_channel
        s_channel = s_channel.reshape(1, -1).to(dtype=torch.float)

        # Fuse scales
        s_group = (s_group.reshape(-1, size_n).contiguous() / s_channel).to(
            dtype=torch.half
        )
    else:
        max_q_val = 2 ** (num_bits - 1) - 1

        # Compute scale for each channel
        s_channel = torch.max(torch.abs(w), 0, keepdim=True)[0]
        s_channel /= max_q_val

        # Quantize
        q_w = torch.round(w / s_channel).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)
        # Compute ref (dequantized)
        w_ref = q_w.half() * s_channel

        s_group = torch.tensor([], dtype=torch.half)
        # div 2 ** (8 - self.bits)) to offset right shift in unpacking
        s_channel /= 2 ** (8 - num_bits)
        s_channel = s_channel.reshape(-1, size_n).contiguous().to(torch.float)

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        s_group.to(device=orig_device),
        s_channel.to(device=orig_device),
    )
