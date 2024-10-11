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


def check_parity_no_mp(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    precompute: bool = False,
    config: Optional[Float8LinearConfig] = None,
    compile_transformer_block: bool = False,
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

        test_cls.assertEqual(losses[0], losses[1], msg = f"iter: {iter_idx}, loss-ref: {losses[0]}, loss-fp8: {losses[1]}")


def check_parity_bf16_mp(
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
        test_cls.assertEqual(losses[0], losses[1], msg = f"iter: {iter_idx}, loss-ref: {losses[0]}, loss-fp8: {losses[1]}")
