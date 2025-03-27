import pytest
import torch
from torchao.prototype.grouped_mm import grouped_mm
from torchao.float8.config import Float8LinearRecipeName


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_fast_accum", [True, False])
@pytest.mark.parametrize("strided", [True, False])
def test_grouped_gemm_2d_3d(use_fast_accum, strided):
    device = "cuda"
    s_int = int(strided)
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(m * n_groups, k * (1 + s_int), device=device)[:, :k]
    b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device)[::(1 + s_int), :, :k]
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
    result = grouped_mm(
        a, b,
        offs=offs, 
        float8_recipe=Float8LinearRecipeName.ROWWISE, 
        out_dtype=torch.bfloat16, 
        use_fast_accum=use_fast_accum
    )
    assert isinstance(result, torch.Tensor)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_fast_accum", [True, False])
@pytest.mark.parametrize("strided", [True, False])
def test_grouped_gemm_3d_3d(use_fast_accum, strided):
    device = "cuda"
    s_int = int(strided)
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device)[::(1 + s_int), :, :k]
    b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device)[::(1 + s_int), :, :k]
    result = grouped_mm(
        a, b,
        float8_recipe=Float8LinearRecipeName.ROWWISE, 
        out_dtype=torch.bfloat16, 
        use_fast_accum=use_fast_accum
    )
    assert isinstance(result, torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensorwise_scaling_not_supported():
    device = "cuda"
    m, n, k, n_groups = 16, 32, 16, 4
    a = torch.randn(m * n_groups, k, device=device)[:, :k]
    b = torch.randn(n_groups, n, k, device=device)[::1, :, :k]
    offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        result = grouped_mm(
            a, b,
            offs=offs, 
            float8_recipe=Float8LinearRecipeName.TENSORWISE, 
            out_dtype=torch.bfloat16, 
        )
