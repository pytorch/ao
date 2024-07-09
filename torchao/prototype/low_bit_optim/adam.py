from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.distributed._tensor import DTensor

from .subclass_8bit import maybe_new_8bit_zero_buffer
from .subclass_4bit import maybe_new_4bit_zero_buffer
from .subclass_fp8 import maybe_new_fp8_zero_buffer


class _Adam(Optimizer):
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

    @staticmethod
    def _new_buffer(p: Tensor, signed: bool, block_size: int):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                # unwrap DTensor
                # set requires_grad for unwrapped param to avoid torch.compile() recompilation 
                # if isinstance(p, DTensor):
                #     p = p._local_tensor.requires_grad_(True)
                # if isinstance(grad, DTensor):
                #     grad = grad._local_tensor

                state = self.state[p]

                # uncomment this to flatten tensor to avoid recompiling
                # but you will get the following error
                # AssertionError: s8 (could be from ["L['grad']._base._local_tensor.size()[0]"]) not in {s3: ["L['exp_avg']._local_tensor.scale.size()[0]", "L['exp_avg']._local_tensor.scale.size()[0]", "L['exp_avg']._local_tensor.scale.size()[0]", "L['exp_avg']._local_tensor.scale.size()[0]", "L['exp_avg']._local_tensor.scale.size()[0]"], s4: ["L['exp_avg']._local_tensor.qmap.size()[0]", "L['exp_avg']._local_tensor.qmap.size()[0]", "L['exp_avg']._local_tensor.qmap.size()[0]", "L['exp_avg']._local_tensor.qmap.size()[0]", "L['exp_avg']._local_tensor.qmap.size()[0]", "L['exp_avg_sq']._local_tensor.qmap.size()[0]", "L['exp_avg_sq']._local_tensor.qmap.size()[0]", "L['exp_avg_sq']._local_tensor.qmap.size()[0]", "L['exp_avg_sq']._local_tensor.qmap.size()[0]", "L['exp_avg_sq']._local_tensor.qmap.size()[0]"], s12: ["L['grad']._local_tensor.size()[0]", "L['grad']._local_tensor.size()[0]"], s10: ["L['grad']._local_tensor.storage_offset()", "L['grad']._local_tensor.storage_offset()"], s16: ["L['exp_avg_sq']._local_tensor.scale.size()[0]", "L['exp_avg_sq']._local_tensor.scale.size()[0]", "L['exp_avg_sq']._local_tensor.scale.size()[0]", "L['exp_avg_sq']._local_tensor.scale.size()[0]", "L['exp_avg_sq']._local_tensor.scale.size()[0]"], s23: ["L['p']._local_tensor.size()[0]", "L['p']._local_tensor.size()[0]"]}.  If this assert is failing, it could be due to the issue described in https://github.com/pytorch/pytorch/pull/90665
                # p = p.view(-1)
                # grad = grad.view(-1)

                # without it, you hit cache size limit

                # State initialization
                # flatten buffer to avoid torch.compile() recompilation
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    state["exp_avg"] = self._new_buffer(p, True, self.block_size)
                    state["exp_avg_sq"] = self._new_buffer(p, False, self.block_size)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = self._new_buffer(p, False, self.block_size)

                state["step"] += 1

                # must explicitly convert lr to Tensor since torch.compile() will treat it as a constant
                # if it is a python float. practically, only lr is changed during training.
                # NOTE: if lr is change at every step, moving lr to CUDA will be a bottleneck.
                if not isinstance(group["lr"], Tensor):
                    group["lr"] = torch.tensor(group["lr"], device=p.device)

                # flatten p and grad to avoid torch.compile() recompilation
                single_param_adam(
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
                )

        return loss


# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
@torch.compile(fullgraph=True, dynamic=True)
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
):
    if weight_decay != 0:
        grad = grad.add(p, alpha=weight_decay)

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


class Adam8bit(_Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        block_size=2048
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

    _new_buffer = staticmethod(maybe_new_8bit_zero_buffer)


class Adam4bit(_Adam):
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
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

    _new_buffer = staticmethod(maybe_new_4bit_zero_buffer)


class AdamFp8(_Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        block_size=2048
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

    @staticmethod
    def _new_buffer(p: Tensor, signed: bool, block_size: int):
        return maybe_new_fp8_zero_buffer(p, block_size)
