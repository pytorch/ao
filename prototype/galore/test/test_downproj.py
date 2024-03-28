import logging

import torch
from triton.testing import do_bench

from galore_fused.triton.kernels.matmul import triton_mm_launcher
from galore_fused.utils import TestGaLoreProjector as GaLoreProjector

# Autotune logging
logging.basicConfig(level=logging.INFO)

torch.manual_seed(0)


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


if __name__ == "__main__":
    M, N = grad_shape = (4096, 4096)
    rank = 128
    allow_tf32 = False
    fp8_fast_accum = False
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    dtype = torch.float32

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
    print("Accuracy: ", torch.max(torch.abs(low_rank_ref() - low_rank_tt())))

    # with torch_profiler_context() as prof:
    #     for _ in range(10):
    #         low_rank_ref()
    #         prof.step()

    ref_perf = do_bench(low_rank_ref)
    tt_perf = do_bench(
        low_rank_tt,
    )
    print(
        f"Performance, torch vs triton: {ref_perf:.4f}ms vs {tt_perf:.4f}ms, {ref_perf / tt_perf:1.2f}x"
    )
