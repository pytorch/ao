import random
from typing import Tuple

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.mx_formats.mx_tensor import to_mx


# --- float8 rowwise scaling ---
def torch_to_float8_per_group_colwise(
    A_col_major: torch.Tensor,
    offs: torch.Tensor,
    target_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function converts the 2D input tensor A to a jagged float8 tensor,
    with scales computed along *logical columns* for each group individually,
    where groups are determined based on the offsets.

    For the right operand of a normal scaled GEMM, the rowwise scales are computed over logical columns.
    (i.e., a tensor of (K,N) will have scales of shape (1,N).

    However, for a 2D right operand of a grouped GEMM, these logical columns go through multiple distinct
    groups/subtensors, for which we want to compute scales individually. So we cannot take one set of scales
    along the logical columns and apply it to the entire tensor.

    Instead, we need to compute scales for each subtensor individually. For a tensor of shape (K,N) this results
    in scales of shape (1,N * num_groups).

    Args:
        A (torch.Tensor): The input tensor to be converted to a jagged float8 tensor.

    Returns:
        A tuple containing the jagged float8 tensor and the scales used for the conversion.
    """
    assert A_col_major.ndim == 2, "A must be 2D"

    num_groups = offs.numel()
    A_fp8_col_major = torch.empty_like(A_col_major, dtype=target_dtype)
    A_scales = torch.empty(
        A_fp8_col_major.size(1) * num_groups,
        dtype=torch.float32,
        device=A_fp8_col_major.device,
    )

    start_idx = 0
    next_scale_idx = 0
    for end_idx in offs.tolist():
        # Get the subtensor of A for this group, fetching the next group of rows, with all columns for each.
        subtensor = A_col_major[start_idx:end_idx, :]  # (local_group_size, K)

        # Compute local rowwise scales for this subtensor, which are along logical columns for the right operand.
        subtensor_scales = tensor_to_scale(
            subtensor,
            target_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=0,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )

        # Apply scales to subtensor and convert to float8.
        tensor_scaled = subtensor.to(torch.float32) * subtensor_scales
        float8_subtensor = to_fp8_saturated(tensor_scaled, target_dtype)

        # Store this portion of the resulting float8 tensor and scales.
        A_fp8_col_major[start_idx:end_idx, :] = float8_subtensor
        A_scales[next_scale_idx : next_scale_idx + subtensor_scales.numel()] = (
            subtensor_scales.squeeze()
        )

        # Update start index for next group.
        start_idx = end_idx
        next_scale_idx += subtensor_scales.numel()

    return A_fp8_col_major, A_scales


def torch_to_float8_per_group_rowwise(
    x: torch.Tensor,
    offs: torch.Tensor,
    target_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function converts the 2D input tensor to a jagged float8 tensor,
    with scales computed along *logical rows* for each group individually,
    where groups are determined based on the offsets.

    For a 2D *left* operand of a normal scaled GEMM, the rowwise scales are computed over logical rows.
    (i.e., a tensor of (M,K) will have scales of shape (M,1).

    However, for a 2D left operand of a grouped GEMM, these logical rows go through multiple distinct
    groups/subtensors, for which we want to compute scales individually. So we cannot take one set of scales
    along the logical rows and apply it to the entire tensor.

    Instead, we need to compute scales for each subtensor individually. For a tensor of shape (M,K) this results
    in scales of shape (M * num_groups, 1).

    Args:
        A (torch.Tensor): The input tensor to be converted to a jagged float8 tensor.

    Returns:
        A tuple containing the jagged float8 tensor and the scales used for the conversion.
    """
    assert x.ndim == 2, "input tensor must be 2D"

    num_groups = offs.numel()
    x_fp8 = torch.empty_like(x, dtype=target_dtype)
    x_scales = torch.empty(
        x_fp8.size(0) * num_groups, dtype=torch.float32, device=x_fp8.device
    )

    start_idx = 0
    next_scale_idx = 0
    for end_idx in offs.tolist():
        # Get the subtensor of A for this group, fetching all rows with the next group of rows.
        subtensor = x[:, start_idx:end_idx]  # (M, local_group_size)

        # Compute local rowwise scales for this subtensor, which are along logical rows for the left operand.
        subtensor_scales = tensor_to_scale(
            subtensor,
            target_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )

        # Apply scales to subtensor and convert to float8.
        tensor_scaled = subtensor.to(torch.float32) * subtensor_scales
        float8_subtensor = to_fp8_saturated(tensor_scaled, target_dtype)

        # Store this portion of the resulting float8 tensor and scales.
        x_fp8[:, start_idx:end_idx] = float8_subtensor
        x_scales[next_scale_idx : next_scale_idx + subtensor_scales.numel()] = (
            subtensor_scales.squeeze()
        )

        # Update start index for next group.
        start_idx = end_idx
        next_scale_idx += subtensor_scales.numel()

    return x_fp8, x_scales


def torch_to_3d_rowwise_float8_transpose_rhs(
    input_hp_t: torch.Tensor,  # (E, K, N)
    target_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function converts the 3D input tensor to a float8 tensor, with scales computed along logical columns
    on a per-expert basis.

    Args:
        x (torch.Tensor): The input tensor to be converted to a float8 tensor. Shape (E, K, N).

    Returns:
        A tuple containing the float8 tensor and the scales used for the conversion.
        Output shape: (E, N, K)
        Scales shape: (E, 1, K
    """
    assert _is_column_major(input_hp_t), "input tensor must be column-major"
    input_hp = input_hp_t.transpose(-2, -1)  # (E, N, K)
    scales = tensor_to_scale(
        input_hp,
        target_dtype,
        scaling_granularity=ScalingGranularity.AXISWISE,
        axiswise_dim=-2,
        round_scales_to_power_of_2=round_scales_to_power_of_2,
    )  # (E, 1, K)

    # Apply scales to tensor and convert to float8.
    tensor_scaled = input_hp.to(torch.float32) * scales
    float8_tensor = to_fp8_saturated(tensor_scaled, target_dtype)

    # To column major
    float8_tensor = float8_tensor.transpose(-2, -1).contiguous().transpose(-2, -1)
    return float8_tensor, scales


# --- mxfp8 scaling ---
def _to_mxfp8_per_group_rowwise(
    x: torch.Tensor,
    offs: torch.Tensor,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is a reference implementation used for testing correctness, it is not performant.

    This function converts the 2D input tensor a mxpf8 tensor along dim 0 with per-token-group scaling,
    where groups are determined based on the offsets.

    Args:
        A (torch.Tensor): The input tensor to be converted to a jagged mxfp8 tensor.

    Returns:
        A tuple containing the jagged mxpf8 tensor and the scales used for the conversion.
    """
    assert x.ndim == 2, "input tensor must be 2D"
    assert offs.numel() > 0, "offs must be non-empty"

    x_mx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_scales = None

    start_idx = 0
    for end_idx in offs.tolist():
        # Get the subtensor of A for this group, fetching all rows with the next group of rows.
        subtensor = x[:, start_idx:end_idx]  # (M, local_group_size)

        # Perform mxfp8 conversion on logically distinct subtensor.
        scales, mx_subtensor = to_mx(
            subtensor.contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # Store this portion of the resulting mxfp8 tensor and scales.
        x_mx[:, start_idx:end_idx] = mx_subtensor
        if x_scales is None:
            x_scales = scales.view(torch.uint8)  # Needed to support cat op below
        else:
            x_scales = torch.cat((x_scales, scales.view(torch.uint8)), dim=1)

        # Update start index for next group.
        start_idx = end_idx

    return x_mx, x_scales.view(torch.float8_e8m0fnu)


def _to_mxfp8_per_group_colwise(
    A_col_major: torch.Tensor,  # (K, N)
    offs: torch.Tensor,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is a reference implementation used for testing correctness, it is not performant.

    This function converts the 2D input tensor a mxpf8 tensor along dim 1 with per-token-group scaling,
    where groups are determined based on the offsets.

    Args:
        A (torch.Tensor): The input tensor to be converted to a mxfp8 tensor.

    Returns:
        A tuple containing the mxpf8 tensor and the scales used for the conversion.
    """
    assert A_col_major.ndim == 2, "A must be 2D"
    assert offs.numel() > 0, "offs must be non-empty"

    A_mx = torch.empty_like(A_col_major, dtype=torch.float8_e4m3fn)
    A_scales = None

    start_idx = 0
    for end_idx in offs.tolist():
        # Get the subtensor of A for this group, fetching the next group of rows, with all columns for each.
        subtensor = A_col_major[start_idx:end_idx, :]  # (local_group_size, N)

        # Convert to mxfp8 along dim1, by transposing, converting, and transposing back.
        scales, mx_subtensor = to_mx(
            subtensor.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )
        scales, mx_subtensor = scales.transpose(-2, -1), mx_subtensor.transpose(-2, -1)

        # Store this portion of the resulting mxfp8 tensor and scales.
        A_mx[start_idx:end_idx, :] = mx_subtensor
        if A_scales is None:
            A_scales = scales.view(torch.uint8)  # Needed to support cat op below
        else:
            A_scales = torch.cat((A_scales, scales.view(torch.uint8)), dim=0)

        # Update start index for next group.
        start_idx = end_idx

    return A_mx, A_scales.view(torch.float8_e8m0fnu)


def _is_column_major(x: torch.Tensor) -> bool:
    """
    This function checks if the input tensor is column-major.

    Args:
        x (torch.Tensor): The input tensor to be checked.

    Returns:
        A boolean indicating whether the input tensor is column-major.
    """
    assert x.ndim == 2 or x.ndim == 3, "input tensor must be 2D or 3D"
    return x.stride(-2) == 1 and x.stride(-1) > 1


def generate_jagged_offs(E, M, multiple_of=16, dtype=torch.int32, device="cuda"):
    """
    Utility function for tests and benchmarks.

    Generates a tensor of length E, containing random values divisible by 16,
    from 0 to M, in sorted order, and where the final value in the tensor is always M.
    Args:
        E (int): The length of the tensor.
        M (int): The maximum value in the tensor.
    Returns:
        torch.Tensor: A tensor of length E with the specified properties.
    """
    # Ensure M is divisible by 16
    if M % multiple_of != 0:
        raise ValueError(f"M must be divisible by {multiple_of}")

    # Generate a list of possible values
    possible_values = [i for i in range(multiple_of, M + 1, multiple_of)]

    # If E is larger than the number of possible values, raise an error
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")

    # Randomly select E - 1 values from the possible values (excluding M)
    selected_values = torch.tensor(random.sample(possible_values[:-1], E - 1))

    # Append M to the selected values
    selected_values = torch.cat((selected_values, torch.tensor([M])))

    # Sort the selected values
    selected_values, _ = torch.sort(selected_values)

    return selected_values.to(dtype).to(device)
