import contextlib
from typing import List, Optional

import torchao.float8.config as config

import torch
import torch.distributed as dist
import torch.nn as nn
from torchao.float8.config import Float8LinearConfig, ScalingType
from torchao.float8.float8_linear_utils import (
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp
import os


@contextlib.contextmanager
def enable_profiling(enable=False):
    if not enable:
        torch_profiler = contextlib.nullcontext()
        yield None
    else:
        trace_dir = "./profilers"
        rank = torch.distributed.get_rank()
        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            torch.distributed.barrier()
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)
        warmup, active = 1, 2
        wait = 1
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler


def run_training_loop(
    test_cls,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    steps,
    float8_config: Float8LinearConfig,
    precompute: bool = False,
):
    torch._dynamo.reset()
    losses = []
    param_sums = []
    grad_sums = []
    with enable_profiling(False) as torch_profiler:
        for iter_idx in range(steps):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            loss = model(local_inp).sum()
            losses.append(loss)
            loss.backward()
            # param_sum = torch.stack(list(x.full_tensor().reshape(-1) for x in model.parameters())).sum()
            # grad_sum = torch.stack(list(x.grad.full_tensor().reshape(-1) for x in model.parameters())).sum()
            param_sum = torch.stack(list(x.reshape(-1) for x in model.parameters())).sum()
            grad_sum = torch.stack(list(x.grad.reshape(-1) for x in model.parameters())).sum()
            param_sums.append(param_sum)
            grad_sums.append(grad_sum)
            if linear_requires_sync(float8_config):
                sync_float8_amax_and_scale_history(model)
            optim.step()
            if (
                precompute
                and float8_config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            ):
                precompute_float8_dynamic_scale_for_fsdp(model)
            if torch_profiler:
                torch_profiler.step()
    return losses, param_sums, grad_sums


def compare_numerics(
    test_cls,
    losses1: List[torch.Tensor],
    param_sums1: List[torch.Tensor],
    grad_sums1: List[torch.Tensor],
    losses2: List[torch.Tensor],
    param_sums2: List[torch.Tensor],
    grad_sums2: List[torch.Tensor],
):
    assert len(losses1) == len(losses2)
    steps = len(losses1)
    for i in range(steps):
        torch.equal(losses1[i], losses2[i]), f"loss different at {i}: {losses1[i]} vs {losses2[i]}"
        torch.equal(param_sums1[i], param_sums2[i]), f"param_sum different at {i}: {param_sums1[i]} vs {param_sums2[i]}"
        torch.equal(grad_sums1[i], grad_sums2[i]), f"grad_sum different at {i}: {grad_sums1[i]} vs {grad_sums2[i]}"
    

def check_parity_compile(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    precompute: bool = False,
    config: Optional[Float8LinearConfig] = None,
):
    ref_losses: List[torch.Tensor] = []
    ref_param_sums: List[torch.Tensor] = []
    ref_grad_sums: List[torch.Tensor] = []
    for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):
        torch._dynamo.reset()
        for iter_idx in range(1000):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            loss = model(local_inp).sum()
            loss.backward()

            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(model)
            
            param_sum = torch.stack([param.sum() for param in model.parameters()]).sum()
            grad_sum = torch.stack([param.grad.sum() for param in model.parameters()]).sum()
            if model is ref_model:
                ref_losses.append(loss)
                ref_param_sums.append(param_sum)
                ref_grad_sums.append(grad_sum)
            else:
                assert torch.equal(loss, ref_losses[iter_idx]), f"loss different at {iter_idx}: {loss} vs {ref_losses[iter_idx]}"
                assert torch.equal(param_sum, ref_param_sums[iter_idx]), f"param_sum different at {iter_idx}: {param_sum} vs {ref_param_sums[iter_idx]}"
                assert torch.equal(grad_sum, ref_grad_sums[iter_idx]), f"grad_sum different at {iter_idx}: {grad_sum} vs {ref_grad_sums[iter_idx]}"
            optim.step()
            if (
                model is fsdp_model
                and precompute
                and config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            ):
                precompute_float8_dynamic_scale_for_fsdp(model)


def check_parity_eager_ddp_no_mp(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    precompute: bool = False,
    config: Optional[Float8LinearConfig] = None,
):
    # TODO(before land): reorder args and make config not optional
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model:
                for param in model.parameters():
                    dist.all_reduce(param.grad)
                    param.grad.div_(dist.get_world_size())

            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(model)

            optim.step()
            if (
                model is fsdp_model
                and precompute
                and config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            ):
                precompute_float8_dynamic_scale_for_fsdp(model)

        assert torch.equal(losses[0], losses[1]), f"loss different at {iter_idx}: {losses[0]} vs {losses[1]}"


def check_parity_eager_ddp_bf16_mp(
    test_cls,
    ref_model: nn.Module,
    ref_model_bf16: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
):
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in (
            (ref_model_bf16, ref_optim),
            (fsdp_model, fsdp_optim),
        ):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model_bf16:
                for param_bf16, param_fp32 in zip(
                    ref_model_bf16.parameters(), ref_model.parameters()
                ):
                    dist.all_reduce(param_bf16.grad)
                    param_bf16.grad.div_(dist.get_world_size())
                    param_fp32.grad = param_bf16.grad.float()
                    param_bf16.grad = None
            # TODO(future): add amax syncing once delayed scaling is supported
            optim.step()
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)
        test_cls.assertEqual(losses[0], losses[1])
