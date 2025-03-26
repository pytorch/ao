import pytest
import torch
from torchao.prototype.grouped_mm import grouped_mm
from torchao.float8.config import Float8LinearRecipeName


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("float8_recipe", [Float8LinearRecipeName.TENSORWISE, Float8LinearRecipeName.ROWWISE])
@pytest.mark.parametrize("use_fast_accum", [True, False])
def test_grouped_gemm(float8_recipe, use_fast_accum):
    device = "cuda"
    m, n, k, n_groups = 16, 16, 16, 4
    a = torch.randn(m, k * n_groups + k, device=device)
    b = torch.randn(n, k * n_groups + k, device=device)
    offs = torch.arange(k, n_groups * k + 1, k, device=device, dtype=torch.int32)
    result = grouped_mm(
        a, b.t(), 
        offs=offs, 
        float8_recipe=float8_recipe, 
        out_dtype=torch.bfloat16, 
        use_fast_accum=use_fast_accum
    )
    assert isinstance(result, torch.Tensor)
