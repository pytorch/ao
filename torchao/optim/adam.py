# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch import Tensor
from torch.distributed._tensor import DTensor
from torch.optim import Optimizer

from .quant_utils import _fp32_to_bf16_sr
from .subclass_4bit import OptimState4bit
from .subclass_8bit import OptimState8bit
from .subclass_fp8 import OptimStateFp8


class _AdamBase(Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas,
        eps,
        weight_decay,
        amsgrad,
        *,
        block_size,
        bf16_stochastic_round,
        is_adamw,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=torch.tensor(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)
        self.block_size = block_size
        self.bf16_stochastic_round = bf16_stochastic_round
        self.is_adamw = is_adamw

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    # bring your own function to create zero-filled subclass
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        raise NotImplementedError

    def _new_buffer(self, p: Tensor, signed: bool):
        local_p = p.to_local() if isinstance(p, DTensor) else p

        # follow bitsandbytes, only quantize tensors >= 4096 values
        if local_p.numel() >= 4096 and local_p.numel() % self.block_size == 0:
            out = self._subclass_zeros(local_p, signed, self.block_size)
        else:
            out = torch.zeros_like(local_p)

        # wrap subclass in DTensor as needed
        # NOTE: local tensor may have different shapes across ranks.
        # this happens when the 1st dim is not divisible by WORLD_SIZE.
        # thus, we must supply shape (and stride) to DTensor.from_local()
        if isinstance(p, DTensor):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        # when there is CPU offload, p.device is cpu, but device_mesh.device_type is cuda.
        # DTensor.from_local() will move local_tensor to device_mesh.device_type.
        # hence, we need to manually move it back to CPU.
        # https://github.com/pytorch/pytorch/blob/bc4cf1c1/torch/distributed/tensor/_api.py#L410-L415
        out = out.to(p.device)
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # for a given model, the number of different argument combinations to single_param_adam() is fixed.
        # thus, it is safe to disable cache limit without the risk of always re-compiling.
        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = self._new_buffer(p, False)

                    state["step"] += 1

                    if not isinstance(group["lr"], Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want to update lr, please use "
                            "optim.param_groups[0]['lr'].fill_(new_lr)"
                        )

                    # without calling p.detach(), torch.compile() will have issues with FSDP2 in some cases
                    # https://github.com/pytorch/ao/issues/652#issuecomment-2285040894
                    # thus, by calling p.detach(), DTensor won't have .grad anymore, which is ok since we
                    # are passing grad separately anyway.
                    torch.compile(single_param_adam, fullgraph=True, dynamic=False)(
                        p.detach(),
                        grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state.get("max_exp_avg_sq", None),
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                        self.is_adamw,
                        self.bf16_stochastic_round and p.dtype is torch.bfloat16,
                    )

        return loss


# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
# and param tensor subclass that implements aten.add_.Tensor, and aten.addcdiv_.default
def single_param_adam(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Optional[Tensor],
    lr: Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    IS_ADAMW: bool,
    BF16_STOCHASTIC_ROUND: bool,
):
    # compute in FP32 for accurate calculations
    p_f32 = p.float()
    grad_f32 = grad.float()

    if IS_ADAMW:
        p_f32 = p_f32 - lr * weight_decay * p_f32
    else:
        grad_f32 = grad_f32 + weight_decay * p_f32

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    # keep high precision copy for param update
    exp_avg_f32 = exp_avg.float().lerp(grad_f32, 1 - beta1).float()
    exp_avg_sq_f32 = exp_avg_sq.float().lerp(grad_f32.square(), 1 - beta2)

    exp_avg.copy_(exp_avg_f32)
    exp_avg_sq.copy_(exp_avg_sq_f32)

    if max_exp_avg_sq is not None:
        max_exp_avg_sq_f32 = torch.maximum(max_exp_avg_sq.float(), exp_avg_sq_f32)
        max_exp_avg_sq.copy_(max_exp_avg_sq_f32)
        denom = (max_exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps
    else:
        denom = (exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps

    p_f32 = p_f32 - lr * (exp_avg_f32 / bias_correction1) / denom

    if BF16_STOCHASTIC_ROUND:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


class Adam8bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=False,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)


class Adam4bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        block_size=128,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=False,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState4bit.zeros(p.shape, signed, block_size, p.device)


class AdamFp8(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=False,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimStateFp8.zeros(p.shape, block_size, p.device)


class AdamW8bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)


class AdamW4bit(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=128,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState4bit.zeros(p.shape, signed, block_size, p.device)


class AdamWFp8(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=256,
        bf16_stochastic_round=False,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimStateFp8.zeros(p.shape, block_size, p.device)


class _AdamW(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        bf16_stochastic_round=False,
    ) -> None:
        """AdamW optimizer that supports quantized training (parameter is quantized). This optimizer should
        only be used with torchao's quantized training."""
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=float("inf"),
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )
