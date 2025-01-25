import torch
import triton
from triton.testing import do_bench

from torchao.prototype.galore.kernels.adam_downproj_fused import fused_adam_mm_launcher
from torchao.prototype.galore.kernels.adam_step import triton_adam_launcher
from torchao.prototype.galore.kernels.matmul import triton_mm_launcher
from torchao.prototype.galore.utils import TestGaLoreProjector as GaLoreProjector

torch.manual_seed(0)

BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
STEP_SIZE = 1e-4


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

    return exp_avg, exp_avg2, grad, galore_proj.ortho_matrix, params


def make_copy(*args):
    return [t.detach().clone() for t in args]


def _ref_op(
    grad,
    proj_matrix,
    exp_avg,
    exp_avg2,
    params,
    beta1=BETA1,
    beta2=BETA2,
    eps=EPS,
    step_size=STEP_SIZE,
    **kwargs,
):
    # Step 1: Down proj grad
    M, N = grad.shape
    if M >= N:
        a, b = grad, proj_matrix.t()
    else:
        a, b = proj_matrix.t(), grad
    low_rank_grad = a @ b

    # Step 2: update adam state
    exp_avg.mul_(beta1).add_(low_rank_grad, alpha=(1.0 - beta1))
    exp_avg2.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)
    denom = exp_avg2.sqrt().add_(eps)
    low_rank_norm_grad = exp_avg / denom

    # Step 3: project normalized low rank grad to full rank
    if M >= N:
        a, b = low_rank_norm_grad, proj_matrix
    else:
        a, b = proj_matrix, low_rank_norm_grad
    full_grad_norm = a @ b

    # Finally, update params with updated grad
    params.add_(full_grad_norm, alpha=-step_size)

    return exp_avg, exp_avg2, params


def _tt_hybrid(
    grad,
    proj_matrix,
    exp_avg,
    exp_avg2,
    params,
    store=True,
    step_size=STEP_SIZE,
    fp8_fast_accum=False,
    allow_tf32=False,
):
    M, N = grad.shape
    if M >= N:
        a, b = grad, proj_matrix.t()
    else:
        a, b = proj_matrix.t(), grad
    low_rank_grad = a @ b

    exp_avg, exp_avg2, norm_grad = triton_adam_launcher(
        exp_avg, exp_avg2, low_rank_grad, store=store
    )

    if M >= N:
        a, b = low_rank_grad, proj_matrix
    else:
        a, b = proj_matrix, low_rank_grad
    params = triton_mm_launcher(
        a,
        b,
        epilogue_alpha=-step_size,
        epilogue_source=params,
        allow_tf32=allow_tf32,
        fp8_fast_accum=fp8_fast_accum,
    )
    return exp_avg, exp_avg2, params


def _tt_fused(
    grad,
    proj_matrix,
    exp_avg,
    exp_avg2,
    params,
    store=True,
    step_size=STEP_SIZE,
    fp8_fast_accum=False,
    allow_tf32=False,
):
    M, N = grad.shape

    if M >= N:
        a, b = grad, proj_matrix.t()
    else:
        a, b = proj_matrix.t(), grad
    exp_avg, exp_avg2, low_rank_grad = fused_adam_mm_launcher(
        a,
        b,
        exp_avg=exp_avg,
        exp_avg2=exp_avg2,
        store=store,
        fp8_fast_accum=fp8_fast_accum,
        allow_tf32=allow_tf32,
    )

    if M >= N:
        a, b = low_rank_grad, proj_matrix
    else:
        a, b = proj_matrix, low_rank_grad
    params = triton_mm_launcher(
        a,
        b,
        epilogue_alpha=-step_size,
        epilogue_source=params,
        allow_tf32=allow_tf32,
        fp8_fast_accum=fp8_fast_accum,
    )
    return exp_avg, exp_avg2, params

    # logging.basicConfig(level=logging.INFO)


def get_kernel(kernel):
    if kernel == "ref":
        op = _ref_op
    elif kernel == "ref":
        op = torch.compile(_ref_op, fullgraph=True, mode="max-autotune")
    elif kernel == "hybrid":
        op = _tt_hybrid
    elif kernel == "fused":
        op = _tt_fused
    else:
        raise ValueError(f"Unknown kernel {kernel}")

    return lambda *args, **kwargs: op(*args, **kwargs)


def get_benchmark(
    M, N, dtype, allow_tf32, fp8_fast_accum=False, quantiles=[0.5, 0.2, 0.8]
):
    config = triton.testing.Benchmark(
        x_names=["rank"],  # Argument names to use as an x-axis for the plot
        x_vals=[
            32,
            64,
            128,
            256,
            512,
        ],  # Different possible values for `x_name`
        line_arg="kernel",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["torch", "hybrid", "fused", "compiled"],
        # Label name for the lines
        line_names=["torch", "hybrid", "fused", "compiled"],
        # Line styles
        styles=[("black", "-"), ("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name=f"Adam Kernel Comparison Grad shape: {M}x{N}, dtype: {dtype}, allow_tf32: {allow_tf32}\nMedian times (ms)",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )

    def benchmark(rank, kernel):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        exp_avg, exp_avg2, grad, proj_matrix, params = make_data(M, N, rank, dtype)

        if kernel == "torch":
            ms, min_ms, max_ms = do_bench(
                lambda: _ref_op(
                    grad,
                    proj_matrix,
                    exp_avg,
                    exp_avg2,
                    params,
                ),
                quantiles=quantiles,
            )
        if kernel == "hybrid":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _tt_hybrid(
                    grad,
                    proj_matrix,
                    exp_avg,
                    exp_avg2,
                    params,
                    store=True,
                    allow_tf32=allow_tf32,
                    fp8_fast_accum=fp8_fast_accum,
                ),
                quantiles=quantiles,
            )
        if kernel == "fused":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _tt_fused(
                    grad,
                    proj_matrix,
                    exp_avg,
                    exp_avg2,
                    params,
                    store=True,
                    allow_tf32=allow_tf32,
                    fp8_fast_accum=fp8_fast_accum,
                ),
                quantiles=quantiles,
            )
        if kernel == "compiled":
            compiled_op = torch.compile(_ref_op, fullgraph=True, mode="max-autotune")
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: compiled_op(
                    grad,
                    proj_matrix,
                    exp_avg,
                    exp_avg2,
                    params,
                ),
                quantiles=quantiles,
            )

        return ms, max_ms, min_ms

    return triton.testing.perf_report(config)(benchmark)
