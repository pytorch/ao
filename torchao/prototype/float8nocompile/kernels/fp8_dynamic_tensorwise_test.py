import pytest
import torch
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_tensor import LinearMMConfig
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    triton_hp_tensor_to_float8_dynamic,
)


def test_fp8_triton_hp_tensor_to_float8_dynamic():
    assert torch.cuda.is_available()
    device = "cuda"
    input_bf16 = torch.randn((4, 4), dtype=torch.bfloat16, device=device)
    x_bf16 = input_bf16.clone().detach().to(device)
    y_bf16 = input_bf16.clone().detach().to(device)

    # production implementation
    x_fp8 = hp_tensor_to_float8_dynamic(
        x_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
    )

    # float8nocompile triton implementation
    y_fp8 = triton_hp_tensor_to_float8_dynamic(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
    )

    assert torch.allclose(x_fp8._scale, y_fp8._scale, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x_fp8._data, y_fp8._data, atol=1e-3, rtol=1e-3)
