from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.distributed._tensor import DTensor

from .subclass_8bit import OptimState8bit
from .subclass_4bit import OptimState4bit
from .subclass_fp8 import OptimStateFp8


class _AdamBase(Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, *, block_size, is_adamw) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=torch.tensor(lr), betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.block_size = block_size
        self.is_adamw = is_adamw

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
                out = DTensor.from_local(
                    local_tensor=self._subclass_zeros(p.to_local(), signed, self.block_size),
                    device_mesh=p.device_mesh,
                    placements=p.placements,
                    run_check=False,
                )
            else:
                out = self._subclass_zeros(p, signed, self.block_size)
        else:
            out = torch.zeros_like(p)
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

                    torch.compile(single_param_adam, fullgraph=True, dynamic=False)(
                        p,
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
                    )

        return loss


# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
# and param tensor subclass that implements aten.add_.Tensor, and aten.addcdiv_.default
# NOTE: right now all of our optimizer state subclasses will dequant to FP32, thus adam computation
# will be done in FP32 (not purposely). we should explicitly cast all inputs to FP32 to ensure FP32
# computation. will need to benchmark to ensure no slowdown.
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
    is_adamw: bool,
):
    if not is_adamw:
        grad = grad.add(p, alpha=weight_decay)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

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
    if is_adamw:
        # merge weight decay and param update in a single .add_() to make this work with quantized param
        p.add_(-lr * weight_decay * p - step_size * new_exp_avg / denom)
    else:
        p.addcdiv_(new_exp_avg, denom, value=-step_size)


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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size, is_adamw=False)

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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size, is_adamw=False)

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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size, is_adamw=False)

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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size, is_adamw=True)

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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size, is_adamw=True)

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
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size, is_adamw=True)

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
    ) -> None:
        """AdamW optimizer that supports quantized training (parameter is quantized). This optimizer should
        only be used with torchao's quantized training."""
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=float("inf"), is_adamw=True)
