import torch
from torch import Tensor

from .adam_8bit import Adam8bit, single_param_adam
from .subclass_4bit import maybe_new_4bit_zero_buffer


class Adam4bit(Adam8bit):
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
    ):
        # change default block_size
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, block_size=block_size)

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

                state = self.state[p]

                # State initialization
                # state is flattened so that torch.compile won't recompile for tensors with different ndim
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    state["exp_avg"] = maybe_new_4bit_zero_buffer(p.view(-1), True, self.block_size)
                    state["exp_avg_sq"] = maybe_new_4bit_zero_buffer(p.view(-1), False, self.block_size)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = maybe_new_4bit_zero_buffer(p.view(-1), False, self.block_size)

                state["step"] += 1

                # must explicitly convert lr to Tensor since torch.compile() will treat it as a constant
                # if it is a python float. practically, only lr is changed during training.
                # NOTE: if lr is change at every step, moving lr to CUDA will be a bottleneck.
                if not isinstance(group["lr"], Tensor):
                    group["lr"] = torch.tensor(group["lr"], device=p.device)

                # flatten p and grad so that torch.compile won't recompile for tensors with different ndim
                single_param_adam(
                    p.view(-1),
                    grad.view(-1),
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
