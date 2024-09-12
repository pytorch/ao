import itertools
import os
import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_2

try:
    # Only works for torch2.2 or newer.
    if TORCH_VERSION_AT_LEAST_2_2:
        from torchao.kernel import intmm_triton
    else:
        intmm_triton = None
except ImportError:
    # On cpu-only builds might not be available.
    intmm_triton = None

AUTOTUNER_ENABLE = bool(int(os.getenv("TORCHAO_AUTOTUNER_ENABLE", 0)))

# torch._int_mm doesn't exist before 2.2
if TORCH_VERSION_AT_LEAST_2_2:
    from torch._dynamo import is_compiling as dynamo_is_compiling
    from torch._higher_order_ops.out_dtype import out_dtype
    def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        """
        Performs a safe integer matrix multiplication, considering different paths for
        torch.compile, cublas, and fallback cases.

        Args:
            input (torch.Tensor): The input tensor of shape [i, j].
            mat2 (torch.Tensor): The matrix to multiply with, of shape [j, k].

        Returns:
            torch.Tensor: The result of the matrix multiplication.

        Raises:
            AssertionError: If the tensors are not on the same device.
        """
        # torch.compile path
        if dynamo_is_compiling() or "FakeTensor" in input.__repr__():
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

        # error checking for cublas path
        assert (
            mat2.device == input.device
        ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
        device_cpu = "cpu" in [mat2.device.type, input.device.type]
        # with input.shape = [i,j] and mat2.shape = [j,k]
        i_is_strictly_greater_than_16 = input.shape[0] > 16
        j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
        k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
        bad_dimensions_for_cublas = not (
            i_is_strictly_greater_than_16
            and j_is_nonzero_multiple_of_8
            and k_is_nonzero_multiple_of_8
        )

        if device_cpu or bad_dimensions_for_cublas:
            # fallback path
            return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
                input.device.type
            )

        # cublas paths
        if not mat2.is_contiguous():  # silently gives incorrect result without this
            mat2 = mat2.contiguous()
        if (not input.is_contiguous()) and (
            input.shape[0] % 8 != 0
        ):  # gives cryptic error without this
            input = (
                input.contiguous()
            )  # (it seems the transpose makes cublas check the above j constraint on i)
        return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)
else:
    def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        """
        Performs a fallback integer matrix multiplication for torch versions before 2.2.

        Args:
            input (torch.Tensor): The input tensor of shape [i, j].
            mat2 (torch.Tensor): The matrix to multiply with, of shape [j, k].

        Returns:
            torch.Tensor: The result of the matrix multiplication in int32.
        """
        # We can improve on this by writing Triton code that works for older versions of Triton
        # that ship with 2.1 or 2.0.
        return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(torch.int32)


def int_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs integer matrix multiplication using intmm_triton if available and autotuner is enabled,
    otherwise falls back to safe_int_mm.

    Args:
        a (torch.Tensor): The first matrix to multiply.
        b (torch.Tensor): The second matrix to multiply.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    if intmm_triton is not None and AUTOTUNER_ENABLE:
        return torch.ops.torchao.int_matmul(a, b)
    return safe_int_mm(a, b)


def int_scaled_matmul(
    a: torch.Tensor, b: torch.Tensor, row_scales: torch.Tensor, col_scales: torch.Tensor
) -> torch.Tensor:
    """
    Performs scaled integer matrix multiplication.

    Args:
        a (torch.Tensor): The first matrix to multiply.
        b (torch.Tensor): The second matrix to multiply.
        row_scales (torch.Tensor): The scaling factors for the rows of the result.
            This can be calculated from row-wise scales of a.
        col_scales (torch.Tensor): The scaling factors for the columns of the result.
            This can be calculated from column-wise scales of b.

    Returns:
        torch.Tensor: The result of the scaled matrix multiplication, with dtype of row_scales.

    Raises:
        AssertionError: If the dimensions of the input tensors do not match the expected shapes.
    """
    assert a.dtype is torch.int8 and b.dtype is torch.int8
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    assert row_scales.shape == (M, 1)
    assert row_scales.is_contiguous()
    assert col_scales.shape == (1, N)
    assert col_scales.is_contiguous()

    if intmm_triton is not None and AUTOTUNER_ENABLE:
        return torch.ops.torchao.int_scaled_matmul(a, b, row_scales, col_scales)

    # perform multiplication in FP32 to prevent overflow
    c = safe_int_mm(a, b)
    return c * row_scales * col_scales
