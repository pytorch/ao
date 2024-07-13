import torch
from torch import Tensor
from torch.optim import Optimizer

from .adam import single_param_adam
from .adamw import single_param_adamw
from .quant_utils import quantize_4bit_with_qmap, dequant_with_qmap
from .subclass_4bit import QMAP_SIGNED, QMAP_UNSIGNED


class Adam4bit(Optimizer):
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
            state["packed_4bit"] = torch.zeros_like(p, dtype=torch.uint8)

            n_scale = p.numel() // self.block_size
            state["scale1"] = p.new_zeros(n_scale)
            state["scale2"] = p.new_zeros(n_scale)

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
                    self._init_state(p, state, group)

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
                    state.get("exp_avg", None),
                    state.get("exp_avg_sq", None),
                    state.get("packed_4bit", None),
                    state.get("scale1", None),
                    state.get("scale2", None),
                )
                _group.append(p_grad_state)

            group_and_hparams = (
                _group,
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
    for group, lr, (beta1, beta2), weight_decay, eps, qmap_signed, qmap_unsigned in param_groups:
        for p, grad, step, exp_avg, exp_avg_sq, packed_4bit, scale1, scale2 in group:
            # unpack and dequant
            if packed_4bit is not None:
                exp_avg = dequant_with_qmap(packed_4bit >> 4, qmap_signed, scale1)
                exp_avg_sq = dequant_with_qmap(packed_4bit & 0b1111, qmap_unsigned, scale2)

            single_param_adam(p, grad, step, exp_avg, exp_avg_sq, None, lr, beta1, beta2, weight_decay, eps)

            # quant and re-pack
            if packed_4bit is not None:
                block_size = exp_avg.numel() // scale1.numel()
                exp_avg_codes, new_scale1 = quantize_4bit_with_qmap(exp_avg, qmap_signed, block_size)
                exp_avg_sq_codes, new_scale2 = quantize_4bit_with_qmap(exp_avg_sq, qmap_unsigned, block_size)

                packed_4bit.copy_((exp_avg_codes << 4) | exp_avg_sq_codes)
                scale1.copy_(new_scale1)
                scale2.copy_(new_scale2)


class AdamW4bit(Adam4bit):
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_groups = self._prepare_param_groups()
        param_groups_adamw_4bit(param_groups)

        return loss


# static compile optim step for all params in a single graph
@torch.compile(fullgraph=True)
def param_groups_adamw_4bit(param_groups):
    for group, lr, (beta1, beta2), weight_decay, eps, qmap_signed, qmap_unsigned in param_groups:
        for p, grad, step, exp_avg, exp_avg_sq, packed_4bit, scale1, scale2 in group:
            # unpack and dequant
            if packed_4bit is not None:
                exp_avg = dequant_with_qmap(packed_4bit >> 4, qmap_signed, scale1)
                exp_avg_sq = dequant_with_qmap(packed_4bit & 0b1111, qmap_unsigned, scale2)

            single_param_adamw(p, grad, step, exp_avg, exp_avg_sq, None, lr, beta1, beta2, weight_decay, eps)

            # quant and re-pack
            if packed_4bit is not None:
                block_size = exp_avg.numel() // scale1.numel()
                exp_avg_codes, new_scale1 = quantize_4bit_with_qmap(exp_avg, qmap_signed, block_size)
                exp_avg_sq_codes, new_scale2 = quantize_4bit_with_qmap(exp_avg_sq, qmap_unsigned, block_size)

                packed_4bit.copy_((exp_avg_codes << 4) | exp_avg_sq_codes)
                scale1.copy_(new_scale1)
                scale2.copy_(new_scale2)
