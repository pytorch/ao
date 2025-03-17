# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from torchao.float8.config import (
    Float8LinearConfig,
    ScalingType,
)
from torchao.float8.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp


def check_parity_no_mp(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    config: Float8LinearConfig,
    precompute: bool = False,
    compile_transformer_block: bool = False,
):
    # check that requires_grad matches ref module
    for ref_param, fsdp_param in zip(ref_model.parameters(), fsdp_model.parameters()):
        test_cls.assertEqual(
            ref_param.requires_grad,
            fsdp_param.requires_grad,
            msg=f"ref_param.requires_grad: {ref_param.requires_grad}, fsdp_param.requires_grad: {fsdp_param.requires_grad}",
        )

    # TODO(before land): reorder args and make config not optional
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model:
                for param in model.parameters():
                    if param.requires_grad:
                        dist.all_reduce(param.grad)
                        param.grad.div_(dist.get_world_size())

            optim.step()
            if (
                model is fsdp_model
                and precompute
                and config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            ):
                precompute_float8_dynamic_scale_for_fsdp(model)

        test_cls.assertEqual(
            losses[0],
            losses[1],
            msg=f"iter: {iter_idx}, loss-ref: {losses[0]}, loss-fp8: {losses[1]}",
        )


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
            optim.step()
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)
        test_cls.assertEqual(
            losses[0],
            losses[1],
            msg=f"iter: {iter_idx}, loss-ref: {losses[0]}, loss-fp8: {losses[1]}",
        )
