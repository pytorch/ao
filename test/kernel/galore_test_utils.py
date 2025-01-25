import torch

from torchao.prototype.galore.kernels.adam_downproj_fused import fused_adam_mm_launcher
from torchao.prototype.galore.kernels.adam_downproj_fused import (
    set_tuner_top_k as adam_downproj_tuner_topk,
)
from torchao.prototype.galore.kernels.adam_step import triton_adam_launcher
from torchao.prototype.galore.kernels.matmul import set_tuner_top_k as matmul_tuner_topk
from torchao.prototype.galore.kernels.matmul import triton_mm_launcher
from torchao.prototype.galore.utils import TestGaLoreProjector as GaLoreProjector

torch.manual_seed(0)

adam_downproj_tuner_topk(10)
matmul_tuner_topk(10)

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
