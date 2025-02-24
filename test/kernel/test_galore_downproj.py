import pytest

# Skip entire test if triton is not available, otherwise CI failure
try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("triton is not installed", allow_module_level=True)

import torch
from galore_test_utils import make_data

from torchao.prototype.galore.kernels.matmul import set_tuner_top_k as matmul_tuner_topk
from torchao.prototype.galore.kernels.matmul import triton_mm_launcher
from torchao.testing.utils import skip_if_rocm

torch.manual_seed(0)

matmul_tuner_topk(10)
MAX_DIFF_no_tf32 = 1e-4
MAX_DIFF_tf32 = 1e-2


TEST_CONFIGS = [
    # (4096, 4096, 128, True, False, torch.float32),
    (4096, 4096, 128, False, False, torch.float32),
    # (4096, 11008, 128, True, False, torch.float32),
    (4096, 11008, 128, False, False, torch.float32),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("M, N, rank, allow_tf32, fp8_fast_accum, dtype", TEST_CONFIGS)
@skip_if_rocm("ROCm enablement in progress")
def test_galore_downproj(M, N, rank, allow_tf32, fp8_fast_accum, dtype):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    MAX_DIFF = MAX_DIFF_tf32 if allow_tf32 else MAX_DIFF_no_tf32
    exp_avg, exp_avg2, grad, galore_proj, params = make_data(M, N, rank, dtype)

    if M >= N:
        a, b = grad, galore_proj.t()
    else:
        a, b = galore_proj.t(), grad
    low_rank_ref = lambda: a @ b
    low_rank_tt = lambda: triton_mm_launcher(
        a, b, allow_tf32=allow_tf32, fp8_fast_accum=fp8_fast_accum
    )
    diff = torch.max(torch.abs(low_rank_ref() - low_rank_tt()))
    if not diff < MAX_DIFF:
        print("diff: ", torch.max(torch.abs(low_rank_ref() - low_rank_tt())))
    assert diff < MAX_DIFF
