import pytest
import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor


def quantize(x, swizzle=False):
    return MXTensor.to_mx(x, torch.float8_e4m3fn, 32, is_swizzled_scales=swizzle)


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
@pytest.mark.parametrize("swizzle", [False, True])
@pytest.mark.parametrize(
    "rows,width,dim", [((128, 256), 128, 0), ((127, 2, 129), 160, -2)]
)
def test_cat(device, swizzle, rows, width, dim):
    if device != "cpu" and not getattr(torch, device).is_available():
        pytest.skip(f"{device} unavailable")
    inputs = [torch.randn(n, width, device=device, dtype=torch.bfloat16) for n in rows]
    result = torch.cat([quantize(x, swizzle) for x in inputs], dim=dim)
    expected = quantize(torch.cat(inputs), swizzle)
    assert isinstance(result, MXTensor)
    assert result.shape == expected.shape
    assert result.is_swizzled_scales == swizzle
    for field in ("qdata", "scale"):
        assert torch.equal(
            getattr(result, field).view(torch.uint8),
            getattr(expected, field).view(torch.uint8),
        )


def test_cat_rejects_incompatible_weights():
    first = quantize(torch.randn(2, 32, dtype=torch.bfloat16))
    second = quantize(torch.randn(3, 32, dtype=torch.bfloat16))
    second.orig_dtype = torch.float32
    with pytest.raises(ValueError, match="matching quantization metadata"):
        torch.cat([first, second])
    with pytest.raises(TypeError, match="only MXTensor"):
        torch.cat([torch.randn(2, 32), first])
    with pytest.raises(NotImplementedError, match="contiguous qdata"):
        torch.cat([first.t(), first.t()])
    with pytest.raises(NotImplementedError, match="dim=0"):
        torch.cat([first, first], dim=1)
