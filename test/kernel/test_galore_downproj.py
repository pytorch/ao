import pytest
import torch
from triton.testing import do_bench

from torchao.prototype.galore.kernels.matmul import triton_mm_launcher
from torchao.prototype.galore.utils import TestGaLoreProjector as GaLoreProjector

torch.manual_seed(0)
MAX_DIFF_no_tf32 = 1e-4

MAX_DIFF_tf32 = 1e-2


def make_data(M, N, rank, dtype):
    grad = torch.randn(M, N, device="cuda", dtype=dtype)
    params = torch.randn(M, N, device="cuda", dtype=dtype)

    galore_proj = GaLoreProjector(rank=rank)
    galore_proj.update_orthogonal_matrix(grad)

    if M >= N:
        exp_avg = torch.randn(M, rank, device="cuda", dtype=dtype)
    else:
        exp_avg = torch.randn(rank, N, device="cuda", dtype=dtype)
    exp_avg2 = exp_avg**2

    return exp_avg, exp_avg2, grad, galore_proj, params


TEST_CONFIGS = [
    (4096, 4096, 128, True, False, torch.float32),
    (4096, 4096, 128, False, False, torch.float32),
    (4096, 11008, 128, True, False, torch.float32),
    (4096, 11008, 128, False, False, torch.float32),
]


@pytest.mark.parametrize("M, N, rank, allow_tf32, fp8_fast_accum, dtype", TEST_CONFIGS)
def test_galore_downproj(M, N, rank, allow_tf32, fp8_fast_accum, dtype):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    MAX_DIFF = MAX_DIFF_tf32 if allow_tf32 else MAX_DIFF_no_tf32
    exp_avg, exp_avg2, grad, galore_proj, params = make_data(M, N, rank, dtype)
    print(
        f"Running with {M} x {N} grad / param, GaLore orthogonal matrix {list(galore_proj.ortho_matrix.shape)}, and {dtype}"
    )

    if M >= N:
        a, b = grad, galore_proj.ortho_matrix.t()
    else:
        a, b = galore_proj.ortho_matrix.t(), grad
    low_rank_ref = lambda: a @ b
    low_rank_tt = lambda: triton_mm_launcher(
        a, b, allow_tf32=allow_tf32, fp8_fast_accum=fp8_fast_accum
    )
    diff = torch.max(torch.abs(low_rank_ref() - low_rank_tt()))
    if not diff < MAX_DIFF:
        print("diff: ", torch.max(torch.abs(low_rank_ref() - low_rank_tt())))
    assert diff < MAX_DIFF
