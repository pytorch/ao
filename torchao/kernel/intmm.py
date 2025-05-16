# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_2, check_cpu_version

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    # Only works for torch2.2 or newer.
    if TORCH_VERSION_AT_LEAST_2_2:
        from torchao.kernel import intmm_triton
    else:
        intmm_triton = None
except ImportError:
    logger.warning(
        "Warning: Detected no triton, on systems without Triton certain kernels will not work"
    )
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
            if input.device.type == "cpu":
                # Matmul in int32 is slow on CPU and not supported well by Inductor cpp backend
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, input.float(), mat2.float()
                )
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

        # error checking for cublas path
        assert mat2.device == input.device, (
            f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
        )
        device_cpu = "cpu" in [mat2.device.type, input.device.type]
        # with input.shape = [i,j] and mat2.shape = [j,k]
        j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
        k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
        bad_dimensions_for_cublas = not (
            j_is_nonzero_multiple_of_8 and k_is_nonzero_multiple_of_8
        )

        if device_cpu or bad_dimensions_for_cublas:
            # fallback path
            return torch.matmul(
                input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)
            ).to(input.device.type)

        # cublas paths
        if not mat2.is_contiguous():  # silently gives incorrect result without this
            mat2 = mat2.contiguous()
        if (not input.is_contiguous()) and (
            input.shape[0] % 8 != 0
        ):  # gives cryptic error without this
            input = (
                input.contiguous()
            )  # (it seems the transpose makes cublas check the above j constraint on i)
        try:
            return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)
        except Exception:
            # fallback path, would run on H100 for float8 dtypes
            # Exception on H100 float8 dtype : "addmm_cuda" not implemented for 'Float8_e4m3fn'
            return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(
                torch.int32
            )
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
        return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(
            torch.int32
        )


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
    a: torch.Tensor, b: torch.Tensor, scales1: torch.Tensor
) -> torch.Tensor:
    """
    Performs scaled integer matrix multiplication.

    Args:
        a (torch.Tensor): The first matrix to multiply.
        b (torch.Tensor): The second matrix to multiply.
        scales1 (torch.Tensor): The scaling factors for the rows of the result.

    Returns:
        torch.Tensor: The result of the scaled matrix multiplication.

    Raises:
        AssertionError: If the dimensions of the input tensors do not match the expected shapes.
    """
    M, K = a.shape
    K, N = b.shape
    assert M == scales1.size(0) or scales1.numel() == 1
    assert 1 == scales1.size(1)
    assert scales1.is_contiguous()
    scales1 = scales1.expand((M, N))
    assert scales1.dim() == 2

    if check_cpu_version(scales1.device):
        # CPU prefers decomposed version of int_scaled_matmul
        # to leverage the fusion capability of Inductor
        c = torch._int_mm(a, b)
        return c.to(scales1.dtype) * scales1

    if intmm_triton is not None and AUTOTUNER_ENABLE:
        return torch.ops.torchao.int_scaled_matmul(a, b, scales1)

    c = safe_int_mm(a, b)
    return c * scales1
