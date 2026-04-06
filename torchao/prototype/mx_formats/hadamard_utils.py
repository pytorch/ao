"""RHT utility functions: Hadamard matrix construction, sign vector helpers, and Triton PIDs.

Provides get_wgrad_sign_vector, get_hadamard_matrix, get_rht_matrix, cast_to_fp4x2,
and the Triton JIT helper _compute_pid.
"""

import functools
import math

import torch
from torch.utils._triton import has_triton


def get_wgrad_sign_vector(
    shape, device, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Generate a random {-1, 1} sign vector for the Hadamard transform."""
    return torch.where(
        torch.rand(shape, device=device) >= 0.5,
        torch.ones(shape, dtype=dtype, device=device),
        -torch.ones(shape, dtype=dtype, device=device),
    )


def get_hadamard_matrix(
    hadamard_dimension: int, device, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix (scaled by 1/sqrt(16))."""
    if hadamard_dimension != 16:
        raise ValueError("Only hadamard dimension 16 is supported.")
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=dtype,
            device=device,
        )
        * hadamard_scale
    )


@functools.lru_cache(maxsize=None)
def get_rht_matrix(
    sign_vector: tuple[int, ...] | None,
    device,
    dtype: torch.dtype = torch.bfloat16,
    hadamard_dimension: int = 16,
) -> torch.Tensor:
    """Construct an RHT matrix from an explicit sign vector or a generated sign vector."""
    if sign_vector is None:
        signs = get_wgrad_sign_vector(hadamard_dimension, device=device, dtype=dtype)
    else:
        if len(sign_vector) != hadamard_dimension:
            raise ValueError(
                f"Expected sign_vector length {hadamard_dimension}, "
                f"got {len(sign_vector)}"
            )
        signs = torch.tensor(sign_vector, dtype=dtype, device=device)
    sign_matrix = signs * torch.eye(hadamard_dimension, dtype=dtype, device=device)
    return sign_matrix @ get_hadamard_matrix(
        hadamard_dimension, device=device, dtype=dtype
    )


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N: tl.constexpr):
        r"""Convert flat tile_id to (pid_n, pid_m) with L2-cache-friendly grouping."""
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % num_pid_in_group) // group_size_n
        return pid_n, pid_m

else:

    def _compute_pid(*args, **kwargs):
        raise RuntimeError("_compute_pid requires Triton")
