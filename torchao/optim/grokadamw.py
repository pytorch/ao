# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Callable, Optional

import torch
from torch import Tensor

from .adam import _AdamBase
from .quant_utils import _fp32_to_bf16_sr
from .subclass_4bit import OptimState4bit
from .subclass_8bit import OptimState8bit
from .subclass_fp8 import OptimStateFp8


class _GrokAdamWBase(_AdamBase):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size,
        bf16_stochastic_round=False,
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0,
    ) -> None:
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError("Invalid alpha_init value: {}".format(alpha_init))
        if lamb < 0.0:
            raise ValueError("Invalid lamb value: {}".format(lamb))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if grokking_signal_decay_rate < 0.0:
            raise ValueError(
                "Invalid grokking_signal_decay_rate value: {}".format(
                    grokking_signal_decay_rate
                )
            )
        if gradient_clipping is not None and gradient_clipping < 0.0:
            raise ValueError(
                "Invalid gradient_clipping value: {}".format(gradient_clipping)
            )

        self._alpha_init = alpha_init
        self._lamb = lamb
        self._gamma = gamma
        self._grokking_signal_decay_rate = grokking_signal_decay_rate
        self._gradient_clipping = gradient_clipping
        self.grokking_signal_fns = grokking_signal_fns

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

        for group in self.param_groups:
            self._set_grok_defaults(group)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        self._set_grok_defaults(self.param_groups[-1])

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            self._set_grok_defaults(group)

    def _set_grok_defaults(self, group: dict) -> None:
        group.setdefault("alpha_init", self._alpha_init)
        group.setdefault("lamb", self._lamb)
        group.setdefault("gamma", self._gamma)
        group.setdefault(
            "grokking_signal_decay_rate", self._grokking_signal_decay_rate
        )
        group.setdefault("gradient_clipping", self._gradient_clipping)
        self._validate_grok_group(group)

    @staticmethod
    def _validate_grok_group(group: dict) -> None:
        if not 0.0 <= group["alpha_init"] <= 1.0:
            raise ValueError("Invalid alpha_init value: {}".format(group["alpha_init"]))
        if group["lamb"] < 0.0:
            raise ValueError("Invalid lamb value: {}".format(group["lamb"]))
        if not 0.0 <= group["gamma"] < 1.0:
            raise ValueError("Invalid gamma value: {}".format(group["gamma"]))
        if group["grokking_signal_decay_rate"] < 0.0:
            raise ValueError(
                "Invalid grokking_signal_decay_rate value: {}".format(
                    group["grokking_signal_decay_rate"]
                )
            )
        if (
            group["gradient_clipping"] is not None
            and group["gradient_clipping"] < 0.0
        ):
            raise ValueError(
                "Invalid gradient_clipping value: {}".format(
                    group["gradient_clipping"]
                )
            )

    def _compute_grokking_signal(self) -> float:
        if not self.grokking_signal_fns:
            return 0.0

        grokking_signal = 0.0
        for grokking_signal_fn in self.grokking_signal_fns:
            value = grokking_signal_fn()
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        "grokking_signal_fns must return scalars, but got shape {}".format(
                            tuple(value.shape)
                        )
                    )
                value = value.item()
            grokking_signal += float(value)

        return grokking_signal / len(self.grokking_signal_fns)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grokking_signal = self._compute_grokking_signal()

        with torch._dynamo.utils.disable_cache_limit():
            for layer_index, group in enumerate(self.param_groups):
                beta1 = group["betas"][0] * (1 - group["gamma"]) ** layer_index
                alpha_t = group["alpha_init"] * math.exp(
                    -group["grokking_signal_decay_rate"] * grokking_signal
                )
                alpha_t = min(max(alpha_t, 0.0), 1.0)

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradient is not supported")

                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)
                        state["grok_ema"] = self._new_buffer(p, True)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = self._new_buffer(p, False)

                    state["step"] += 1

                    if not isinstance(group["lr"], Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want to update lr, please use "
                            "optim.param_groups[0]['lr'].fill_(new_lr)"
                        )

                    torch.compile(single_param_grokadamw, fullgraph=True, dynamic=False)(
                        p.detach(),
                        grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state.get("max_exp_avg_sq", None),
                        state["grok_ema"],
                        group["lr"],
                        beta1,
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                        alpha_t,
                        group["lamb"],
                        0.0
                        if group["gradient_clipping"] is None
                        else group["gradient_clipping"],
                        self.is_adamw,
                        self.bf16_stochastic_round and p.dtype is torch.bfloat16,
                    )

        return loss


def single_param_grokadamw(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Optional[Tensor],
    grok_ema: Tensor,
    lr: Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    alpha_t: float,
    lamb: float,
    gradient_clipping: float,
    IS_ADAMW: bool,
    BF16_STOCHASTIC_ROUND: bool,
):
    p_f32 = p.float()
    grad_f32 = grad.float()

    grok_ema_f32 = grok_ema.float().lerp(grad_f32, 1 - alpha_t)
    grok_ema.copy_(grok_ema_f32)

    grad_f32 = grad_f32 + lamb * grok_ema_f32

    if gradient_clipping > 0:
        clip_coef = torch.clamp(gradient_clipping / (grad_f32.norm() + 1e-6), max=1.0)
        grad_f32 = grad_f32 * clip_coef

    if IS_ADAMW:
        p_f32 = p_f32 - lr * weight_decay * p_f32
    else:
        grad_f32 = grad_f32 + weight_decay * p_f32

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    exp_avg_f32 = exp_avg.float().lerp(grad_f32, 1 - beta1)
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


class GrokAdamW8bit(_GrokAdamWBase):
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
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0,
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
            alpha_init=alpha_init,
            lamb=lamb,
            gamma=gamma,
            grokking_signal_fns=grokking_signal_fns,
            grokking_signal_decay_rate=grokking_signal_decay_rate,
            gradient_clipping=gradient_clipping,
        )
        torch._C._log_api_usage_once("torchao.optim.GrokAdamW8bit")

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(
            p.shape, signed, block_size, p.device, dtype=p.dtype
        )


class GrokAdamW4bit(_GrokAdamWBase):
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
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0,
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
            alpha_init=alpha_init,
            lamb=lamb,
            gamma=gamma,
            grokking_signal_fns=grokking_signal_fns,
            grokking_signal_decay_rate=grokking_signal_decay_rate,
            gradient_clipping=gradient_clipping,
        )
        torch._C._log_api_usage_once("torchao.optim.GrokAdamW4bit")

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState4bit.zeros(
            p.shape, signed, block_size, p.device, dtype=p.dtype
        )


class GrokAdamWFp8(_GrokAdamWBase):
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
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0,
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
            alpha_init=alpha_init,
            lamb=lamb,
            gamma=gamma,
            grokking_signal_fns=grokking_signal_fns,
            grokking_signal_decay_rate=grokking_signal_decay_rate,
            gradient_clipping=gradient_clipping,
        )
        torch._C._log_api_usage_once("torchao.optim.GrokAdamWFp8")

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimStateFp8.zeros(p.shape, block_size, p.device, dtype=p.dtype)
