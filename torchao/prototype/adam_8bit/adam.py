import math

import torch
from torch import Tensor
from torch.optim import Optimizer

from .subclass import DTQ8bit
# from ._optim import AdamInt8


class AdamDTQ8bit(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        block_size=2048,
    ):
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
            group.setdefault('amsgrad', False)

    # follow bitsandbytes
    # only apply quantization for tensor with more than 4096 values
    def _new_zero_buffer(self, p: Tensor, signed: bool = True):
        out = torch.zeros_like(p)
        if p.numel() >= 4096 and p.numel() % self.block_size == 0:
            out = DTQ8bit.from_float(out, signed=signed, block_size=self.block_size)
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0)
                    state['exp_avg'] = self._new_zero_buffer(p, signed=True)
                    state['exp_avg_sq'] = self._new_zero_buffer(p, signed=False)
                    if amsgrad:
                        state['max_exp_avg_sq'] = self._new_zero_buffer(p, signed=False)

                single_adam_step(
                    p,
                    grad,
                    state['step'],
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state.get('max_exp_avg_sq', None),
                    group['lr'],
                    group['betas'][0],
                    group['betas'][1],
                    group['eps'],
                    group['amsgrad'],
                )

        return loss


def single_adam_step(
    p: Tensor,
    grad: Tensor,
    # optim state
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Tensor | None,
    # optim hparams
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool,
):
    step += 1
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # exp_avg.lerp_(grad, 1 - beta1)
    # exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    new_exp_avg = exp_avg.lerp(grad, 1 - beta1)
    new_exp_avg_sq = exp_avg_sq.lerp(grad.square(), 1 - beta2)

    if amsgrad:
        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    else:
        # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        denom = (new_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    # p.addcdiv_(exp_avg, denom, value=-step_size)
    p.addcdiv_(new_exp_avg, denom, value=-step_size)

    exp_avg.copy_(new_exp_avg)
    exp_avg_sq.copy_(new_exp_avg_sq)
