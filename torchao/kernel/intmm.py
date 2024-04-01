import itertools
import os
import torch

from torchao.quantization.utils import TORCH_VERSION_AFTER_2_2

try:
    # Only works for torch2.2 or newer.
    if TORCH_VERSION_AFTER_2_2:
        from torchao.kernel import intmm_triton
    else:
        intmm_triton = None
except ImportError:
    # On cpu-only builds might not be available.
    intmm_triton = None

AUTOTUNER_ENABLE = bool(int(os.getenv("TORCHAO_AUTOTUNER_ENABLE", 0)))

# torch._int_mm doesn't exist before 2.2
if TORCH_VERSION_AFTER_2_2:
    from torch._dynamo import is_compiling as dynamo_is_compiling
    from torch._higher_order_ops.out_dtype import out_dtype
    def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
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
        # We can improve on this by writing Triton code that works for older versions of Triton
        # that ship with 2.1 or 2.0.
        return torch.matmul(input.to(torch.float32), mat2.to(torch.float32)).to(torch.int32)


def int_matmul(a, b):
    if intmm_triton is not None and AUTOTUNER_ENABLE:
        return torch.ops.torchao.int_matmul(a, b)
    return safe_int_mm(a, b)


def int_scaled_matmul(a, b, scales1):
    M, K = a.shape
    K, N = b.shape
    assert M == scales1.size(0)
    assert 1 == scales1.size(1)
    assert scales1.is_contiguous()
    scales1 = scales1.expand((M, N))
    assert scales1.dim() == 2
    if intmm_triton is not None and AUTOTUNER_ENABLE:
        return torch.ops.torchao.int_scaled_matmul(a, b, scales1)

    c = safe_int_mm(a, b)
    return c * scales1
