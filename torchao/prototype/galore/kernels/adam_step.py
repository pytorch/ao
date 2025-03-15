# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl
from triton.language.math import sqrt
from triton.runtime.autotuner import heuristics

from .custom_autotune import Config, autotune

BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8

AUTOTUNER_TOP_K = 10


def get_configs_for_adam(num_warps=[2, 4, 8], block_sizes=[512, 1024, 2048]):
    configs = []
    for w in num_warps:
        for bs in block_sizes:
            configs.append(Config({"BLOCK_SIZE": bs}, num_warps=w))
    return configs


def early_adam_prune(configs, named_args):
    numels = named_args["numels"]
    pruned_configs = [cfg for cfg in configs if numels % cfg.kwargs["BLOCK_SIZE"] == 0]
    # print("Pruned configs:\n")
    for cfg in pruned_configs:
        print(f"{cfg}\n")
    return pruned_configs


def get_adam_tuner(
    configs=get_configs_for_adam(),
    early_config_prune=None,  # early_adam_prune,
    top_k=AUTOTUNER_TOP_K,
):
    return autotune(
        configs=configs,
        prune_configs_by={
            "early_config_prune": early_config_prune,
            "top_k": top_k,
        },
        key=["numels"],
    )


def get_adam_heuristics():
    return {
        "USE_MASK": lambda args: args["numels"] % args["BLOCK_SIZE"] != 0,
    }


@autotune(configs=get_configs_for_adam(), key=["numels"])
@heuristics(get_adam_heuristics())
@triton.jit
def _adam_update(
    avg_ptr,
    avg2_ptr,
    grad_ptr,
    # avg_out_ptr,
    # avg2_out_ptr,
    # grad_out_ptr,
    numels,
    store,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
    BETA1: tl.constexpr = BETA1,
    BETA2: tl.constexpr = BETA2,
    EPS: tl.constexpr = EPS,
):
    pid_m = tl.program_id(0)
    offset = pid_m * BLOCK_SIZE
    offset = offset + tl.arange(0, BLOCK_SIZE)
    # load_idx = offset + tl.arange(0, BLOCK_SIZE)
    load_idx = tl.max_contiguous(tl.multiple_of(offset, BLOCK_SIZE), BLOCK_SIZE)

    mask = None
    if USE_MASK:
        mask = load_idx < numels
    avg = tl.load(avg_ptr + load_idx, mask=mask)
    avg2 = tl.load(avg2_ptr + load_idx, mask=mask)
    grad = tl.load(grad_ptr + load_idx, mask=mask)

    avg = BETA1 * avg + (1.0 - BETA1) * grad
    avg2 = BETA2 * avg2 + (1.0 - BETA2) * (grad * grad)

    denom = sqrt(avg2) + EPS
    # denom = tl.sqrt(avg2) + EPS

    norm_grad = avg / denom

    if store:
        tl.store(avg_ptr + load_idx, avg, mask=mask)
        tl.store(avg2_ptr + load_idx, avg2, mask=mask)
        tl.store(grad_ptr + load_idx, norm_grad, mask=mask)
    # tl.store(avg_out_ptr + load_idx, avg, mask=mask)
    # tl.store(avg2_out_ptr + load_idx, avg2, mask=mask)
    # tl.store(grad_out_ptr + load_idx, norm_grad, mask=mask)


adam_update = _adam_update


def triton_adam_launcher(
    avg,
    avg2,
    grad,
    store=True,
    beta1=BETA1,
    beta2=BETA2,
    eps=EPS,
):
    M, N = avg.shape
    # avg_out = torch.empty_like(avg)
    # avg2_out = torch.empty_like(avg2)
    # grad_out = torch.empty_like(grad)

    grid = lambda META: (triton.cdiv(M * N, META["BLOCK_SIZE"]),)
    adam_update[grid](
        avg,
        avg2,
        grad,
        # avg_out,
        # avg2_out,
        # grad_out,
        avg.numel(),
        store=store,
        BETA1=beta1,
        BETA2=beta2,
        EPS=eps,
        # BLOCK_SIZE=1024,
        # USE_MASK=USE_MASK,
    )
    return avg, avg2, grad


def ref_adam_step(exp_avg, exp_avg2, grad, beta1=BETA1, beta2=BETA2, eps=EPS):
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg2 = beta2 * exp_avg2 + (1 - beta2) * torch.square(grad)
    denom = exp_avg2.sqrt() + eps
    norm_grad = exp_avg / denom
    return exp_avg, exp_avg2, norm_grad


def make_data(M, N, rank, dtype):
    # full_grad = torch.randn(M, N, device="cuda", dtype=dtype)
    params = torch.randn(M, N, device="cuda", dtype=dtype)

    if M >= N:
        exp_avg = torch.randn(M, rank, device="cuda", dtype=dtype)
    else:
        exp_avg = torch.randn(rank, N, device="cuda", dtype=dtype)
    exp_avg2 = exp_avg**2
    down_grad = torch.randn_like(exp_avg)

    return exp_avg, exp_avg2, down_grad, params


if __name__ == "__main__":
    M = N = 4096
    rank = 128
    dtype = torch.float32
    exp_avg, exp_avg2, grad, params = make_data(M, N, rank, dtype=dtype)
    exp_avg_copy, exp_avg2_copy, grad_copy = (
        exp_avg.clone(),
        exp_avg2.clone(),
        grad.clone(),
    )
    ref_out = ref_adam_step(exp_avg, exp_avg2, grad)

    # Autotune run -- changes exp_avg, exp_avg2, grad in-place
    _ = triton_adam_launcher(exp_avg, exp_avg2, grad)
    triton_out = triton_adam_launcher(exp_avg_copy, exp_avg2_copy, grad_copy)

    for ref, tt in zip(ref_out, triton_out):
        print(torch.max(torch.abs(ref - tt)))
