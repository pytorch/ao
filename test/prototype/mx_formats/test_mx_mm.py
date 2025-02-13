import pytest
import torch

from torchao.float8.float8_utils import compute_error
from torchao.ops import mx_fp4_bf16, mx_fp8_bf16
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP4, MXTensor
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4, is_sm_at_least_100

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


def run_matrix_test(M: int, K: int, N: int, format) -> float:
    dtype = torch.bfloat16
    device = torch.device("cuda")

    a = torch.rand((M, K), dtype=dtype, device=device)
    b = torch.rand((N, K), dtype=dtype, device=device)

    fmt = torch.float8_e4m3fn if format == "fp8" else DTYPE_FP4
    mx_func = mx_fp8_bf16 if format == "fp8" else mx_fp4_bf16

    a_mx = MXTensor.to_mx(a, fmt, 32)
    b_mx = MXTensor.to_mx(b, fmt, 32)

    a_data = a_mx._data
    b_data = b_mx._data
    assert b_data.is_contiguous()
    b_data = b_data.transpose(-1, -2)

    a_scale = a_mx._scale_e8m0.view(M, K // 32)
    b_scale = b_mx._scale_e8m0.view(N, K // 32)

    a_scale_block = to_blocked(a_scale)
    b_scale_block = to_blocked(b_scale)

    out_hp = a_mx.to_dtype(torch.bfloat16) @ b_mx.to_dtype(torch.bfloat16).transpose(
        -1, -2
    )
    out = mx_func(a_data, b_data, a_scale_block, b_scale_block)

    return compute_error(out_hp, out).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not is_sm_at_least_100(), reason="CUDA capability >= 10.0 required for mxfloat8"
)
@pytest.mark.parametrize(
    "size",
    [
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),  # Small
        (512, 512, 512),
        (768, 768, 768),  # Medium
        (1024, 1024, 1024),
        (8192, 8192, 8192),  # Large
        (128, 256, 384),
        (256, 384, 512),  # Non-square
        (129, 256, 384),
        (133, 512, 528),  # Non-aligned
    ],
    ids=lambda x: f"{x[0]}x{x[1]}x{x[2]}",
)
@pytest.mark.parametrize("format", ["fp8", "fp4"])
def test_matrix_multiplication(size, format):
    M, K, N = size
    sqnr = run_matrix_test(M, K, N, format)
    threshold = 80.0
    assert (
        sqnr >= threshold
    ), f"{format} SQNR {sqnr} below threshold for dims {M}x{K}x{N}"
