import contextlib
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

import torchao.float8.config as config
from torchao.float8.config import (
    Float8CommLinearConfig,
    Float8LinearConfig,
    ScalingType,
)
from torchao.float8.float8_comm_utils import Float8CommWeightWithDynamicCastTensor

# from torchao.float8.float8_comm_utils import Float8CommWeightWithDynamicCastTensor
from torchao.float8.float8_linear_utils import (
    linear_requires_sync,
    sync_float8_amax_and_scale_history,
)
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_tensor import GemmInputRole
from torchao.float8.fsdp_utils import (
    precompute_float8_dynamic_scale_for_fsdp,
    WeightWithDynamicFloat8CastTensor,
)


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

        if compile_transformer_block:
            test_cls.assertEqual(losses[0], losses[1], atol=1e-4, rtol=1e-4, msg = f"iter: {iter_idx}, loss-ref: {losses[0]}, loss-fp8: {losses[1]}")
        else:
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


def check_parity_fp8_comm_only(
    test_cls,
    ref_model: nn.Module,
    ref_optim: torch.optim.Optimizer,
    fsdp_model: nn.Module,
    fsdp_optim: torch.optim.Optimizer,
    local_inp: torch.Tensor,
    precompute: bool = False,
    config: Optional[Float8CommLinearConfig] = None,
    compile: bool = False,
):
    # print("compile", compile)
    # if compile:
    #     hp_tensor_to_float8_dynamic_func = torch.compile(hp_tensor_to_float8_dynamic)
    #     print("compiled")
    # else:
    #     hp_tensor_to_float8_dynamic_func = hp_tensor_to_float8_dynamic
    
    def to_orig(fp8):
        return fp8.to_original_precision()
    if compile:
        to_orig_func = torch.compile(to_orig)
        print("compiled")
    else:
        to_orig_func = to_orig

    ref_model_orig_params = {}

    # TODO(before land): reorder args and make config not optional
    for iter_idx in range(10):
        losses: List[torch.Tensor] = []
        for model, optim in ((ref_model, ref_optim), (fsdp_model, fsdp_optim)):

            # cast parameters to fp8 and cast back, which simulates fp8-all-gather behaviors.
            if model is ref_model:
                ref_model_orig_params = {}
                for name, param in model.named_parameters():
                    tmp = name.split('.')
                    compiled_name = '.'.join(tmp[:2]+['_orig_mod']+tmp[2:])
                    name = compiled_name if compiled_name in fsdp_model.state_dict() else name
                    if not isinstance(fsdp_model.state_dict()[name]._local_tensor, (WeightWithDynamicFloat8CastTensor, Float8CommWeightWithDynamicCastTensor)):
                        continue
                    # print("float tensors", name)
                    orig_param = param.data
                    ref_model_orig_params[name] = orig_param
                    fp8_param = hp_tensor_to_float8_dynamic(
                        orig_param,
                        torch.float8_e4m3fn,
                        None, # mm_linear_config,
                        reduce_amax=True,
                        gemm_input_role=GemmInputRole.WEIGHT,
                    )
                    # print("fp8_param", fp8_param.to_original_precision())
                    param.data = to_orig_func(fp8_param)
                    # print("ref model", name, param.shape, model.state_dict()[name])

            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            losses.append(model(local_inp).sum())
            losses[-1].backward()
            if model is ref_model:
                for name, param in model.named_parameters():
                    if name in ref_model_orig_params:
                        param.data = ref_model_orig_params[name]
                    dist.all_reduce(param.grad)
                    # param.grad.div_(dist.get_world_size())
            

            if model is ref_model:
                for name, param in model.named_parameters():
                    # print("ref param", iter_idx, name, param)
                    # print("ref grad", iter_idx, name, param.grad)
                    continue
            if model is fsdp_model:
                for name, param in model.named_parameters():
                    # print("fsdp param", iter_idx, name, param)
                    # print("fsdp grad", iter_idx, name, param.grad)
                    continue


            if linear_requires_sync(config):
                sync_float8_amax_and_scale_history(model)

            optim.step()
            if (
                model is fsdp_model
                and precompute
                and config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            ):
                precompute_float8_dynamic_scale_for_fsdp(model)

        test_cls.assertEqual(losses[0], losses[1], f"iter: {iter_idx}, loss-ref: {losses[0]}, loss-fp8: {losses[1]}")
