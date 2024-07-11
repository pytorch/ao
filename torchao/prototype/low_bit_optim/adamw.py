from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.distributed._tensor import DTensor

from .subclass_8bit import OptimState8bit
from .subclass_4bit import OptimState4bit
from .subclass_fp8 import OptimStateFp8


class _AdamW(Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, *, block_size) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.block_size = block_size

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    # bring your own function to create zero-filled subclass
    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        raise NotImplementedError

    # follow bitsandbytes, only quantize tensors >= 4096 values
    # also wrap subclass in DTensor when needed
    def _new_buffer(self, p: Tensor, signed: bool):
        if p.numel() >= 4096 and p.numel() % self.block_size == 0:
            if isinstance(p, DTensor):
                out = torch.empty_like(p)
                out._local_tensor = self._subclass_zeros(
                    out._local_tensor,
                    signed,
                    self.block_size,
                )
            else:
                out = self._subclass_zeros(p, signed, self.block_size)
        else:
            out = torch.zeros_like(p)
        return out

    def _prepare_param_groups(self):
        param_groups = []

        for group in self.param_groups:
            _group = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    state["exp_avg"] = self._new_buffer(p, True)
                    state["exp_avg_sq"] = self._new_buffer(p, False)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = self._new_buffer(p, False)

                state["step"] += 1

                # must explicitly convert lr to Tensor since torch.compile() will treat Python float as constant.
                # practically, only lr is changed during training.
                # NOTE: if lr is changed at every step, moving lr to CUDA can slow down training 3-4%.
                if not isinstance(group["lr"], Tensor):
                    group["lr"] = torch.tensor(group["lr"], device=p.device)

                p_grad_state = (
                    p,
                    grad,
                    state["step"],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state.get("max_exp_avg_sq", None),
                )
                _group.append(p_grad_state)

            param_groups.append((_group, group["lr"], group["betas"], group["weight_decay"], group["eps"]))

        return param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_groups = self._prepare_param_groups()
        param_groups_adamw(param_groups)

        return loss


# static compile optim step for all params at once so we don't need dynamic kernel
@torch.compile(fullgraph=True)
def param_groups_adamw(param_groups):
    for group, lr, (beta1, beta2), weight_decay, eps in param_groups:
        for p, grad, step, exp_avg, exp_avg_sq, max_exp_avg_sq in group:
            single_param_adamw(p, grad, step, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, weight_decay, eps)


# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
def single_param_adamw(
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
):
    p.mul_(1 - lr * weight_decay)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # keep high precision copy for param update
    new_exp_avg = exp_avg.lerp(grad, 1 - beta1)
    new_exp_avg_sq = exp_avg_sq.lerp(grad.square(), 1 - beta2)

    exp_avg.copy_(new_exp_avg)
    exp_avg_sq.copy_(new_exp_avg_sq)

    if max_exp_avg_sq is not None:
        new_max_exp_avg_sq = torch.maximum(max_exp_avg_sq, new_exp_avg_sq)
        max_exp_avg_sq.copy_(new_max_exp_avg_sq)
        denom = (new_max_exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
    else:
        denom = (new_exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

    step_size = lr / bias_correction1
    p.addcdiv_(new_exp_avg, denom, value=-step_size)


class AdamW8bit(_AdamW):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=2048
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)


class AdamW4bit(_AdamW):
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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimState4bit.zeros(p.shape, signed, block_size, p.device)


class AdamWFp8(_AdamW):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        block_size=2048
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        return OptimStateFp8.zeros(p.shape, block_size, p.device)
