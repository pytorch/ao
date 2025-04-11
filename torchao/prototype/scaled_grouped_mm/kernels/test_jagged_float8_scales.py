import pytest
import torch

from torchao.prototype.scaled_grouped_mm.kernels.jagged_float8_scales import (
    _amax_axiswise,
    _scale_axiswise,
)

@pytest.mark.parametrize("input_shape", [(1024, 1024), (8, 256, 2048, 2048)])
@pytest.mark.parametrize("axis", [0,1])
def test_axiswise_amaxes(input_shape, axis):
    device = "cuda"
    x = torch.randn(input_shape, dtype=torch.bfloat16, device=device)

    # compute reference axiswise amaxes
    ref_amaxes = torch.max(torch.abs(x), axis=axis)

    # compute actual axiswise amaxes
    amaxes_size = x.shape[axis]
    amaxes_buffer = torch.empty(amaxes_size, dtype=torch.bfloat16, device=device)
    triton_amaxes = _amax_axiswise(
        x,
        amaxes_buffer,
        x.stride(0),
        x.stride(1),
        axis,
        x.numel(),
        x.dtype,
    )

    assert torch.eq(ref_amaxes, triton_amaxes),f"axiswise amaxes of shape {input_shape} along axis {axis} did not match"
