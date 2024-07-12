import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.distributed._tensor import DTensor

from .adam import single_param_adam
from .subclass_4bit import QMAP_SIGNED, QMAP_UNSIGNED, quantize_4bit_with_qmap


class Adam4bit(Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, amsgrad, *, block_size) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if amsgrad:
            raise ValueError(f"{self.__class__.__name__} does not support amsgrad=True")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.block_size = block_size

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def _init_state(self, p: Tensor, state, group):
        state["step"] = torch.tensor(0.0, device=p.device)

        # follow bitsandbytes, only quantize tensors >= 4096 values
        # also wrap subclass in DTensor when needed
        if p.numel() >= 4096 and p.numel() % self.block_size == 0:
            if isinstance(p, DTensor):
                raise NotImplementedError
            else:
                n_scale = p.numel() // self.block_size
                state["packed_4bit"] = torch.zeros(p.shape, dtype=torch.uint8, device=p.device)
                state["scale1"] = torch.zeros(n_scale, dtype=p.dtype, device=p.device)
                state["scale2"] = torch.zeros(n_scale, dtype=p.dtype, device=p.device)

            # shared qmap among params within a param group
            if "qmap_signed" not in group:
                group["qmap_signed"] = torch.tensor(QMAP_SIGNED, device=p.device)
                group["qmap_unsigned"] = torch.tensor(QMAP_UNSIGNED, device=p.device)

        else:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

    def _prepare_param_groups(self):
        param_groups = []

        for group in self.param_groups:
            _group_4bit = []
            _group_other = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    self._init_state(p, state, group)

                state["step"] += 1

                # must explicitly convert lr to Tensor since torch.compile() will treat Python float as constant.
                # practically, only lr is changed during training.
                # NOTE: if lr is changed at every step, moving lr to CUDA can slow down training 3-4%.
                if not isinstance(group["lr"], Tensor):
                    group["lr"] = torch.tensor(group["lr"], device=p.device)

                if "packed_4bit" in state:
                    p_grad_state = (
                        p,
                        grad,
                        state["step"],
                        state["packed_4bit"],
                        state["scale1"],
                        state["scale2"],
                    )
                    _group_4bit.append(p_grad_state)
                
                else:
                    p_grad_state = (
                        p,
                        grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                    )
                    _group_other.append(p_grad_state)

            group_and_hparams = (
                _group_4bit,
                _group_other,
                group["lr"],
                group["betas"],
                group["weight_decay"],
                group["eps"],
                group.get("qmap_signed", None),
                group.get("qmap_unsigned", None),
            )
            param_groups.append(group_and_hparams)

        return param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_groups = self._prepare_param_groups()
        param_groups_adam_4bit(param_groups)

        return loss


# static compile optim step for all params in a single graph
@torch.compile(fullgraph=True)
def param_groups_adam_4bit(param_groups):
    for group_4bit, group_other, lr, (beta1, beta2), weight_decay, eps, qmap_signed, qmap_unsigned in param_groups:
        for p, grad, step, packed_4bit, scale1, scale2 in group_4bit:
            exp_avg = qmap_signed[(packed_4bit >> 4).int()]
            exp_avg_sq = qmap_unsigned[(packed_4bit & 0b1111).int()]

            exp_avg = exp_avg.view(scale1.shape[0], -1) * scale1.view(-1, 1)
            exp_avg_sq = exp_avg_sq.view(scale2.shape[0], -1) * scale2.view(-1, 1)

            single_param_adam(p, grad, step, exp_avg.view(p.shape), exp_avg_sq.view(p.shape), None, lr, beta1, beta2, weight_decay, eps)

            block_size = exp_avg.numel() // scale1.numel()
            new_exp_avg_codes, new_scale1 = quantize_4bit_with_qmap(exp_avg, qmap_signed, block_size, pack=False)
            new_exp_avg_sq_codes, new_scale2 = quantize_4bit_with_qmap(exp_avg_sq, qmap_unsigned, block_size, pack=False)

            packed_4bit.copy_((new_exp_avg_codes << 4) | new_exp_avg_sq_codes)
            scale1.copy_(new_scale1)
            scale2.copy_(new_scale2)

        for p, grad, step, exp_avg, exp_avg_sq in group_other:
            single_param_adam(p, grad, step, exp_avg, exp_avg_sq, None, lr, beta1, beta2, weight_decay, eps)
