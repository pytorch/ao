import torch
from torch import Tensor

from .adam_8bit import Adam8bit, single_param_adam
from .subclass_4bit import Optim2State4bit


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
        assert not amsgrad, "amsgrad is not supported"
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

                    # https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/config.py#L37
                    # only apply quantization for tensor with more than 4096 values
                    # TODO: also skip 1D tensor? e.g. biases and norm scales
                    if p.numel() >= 4096 and p.numel() % self.block_size == 0:
                        state["two_states"] = Optim2State4bit.zeros(p.view(-1).shape, self.block_size, device=p.device)
                    else:
                        state["exp_avg"] = torch.zeros_like(p.view(-1))
                        state["exp_avg_sq"] = torch.zeros_like(p.view(-1))

                state["step"] += 1

                # flatten p and grad so that torch.compile won't recompile for tensors with different ndim
                # must explicitly convert lr to Tensor since torch.compile() will treat it as a constant
                # if it is a python float. practically, only lr is changed during training.
                # NOTE: if lr is change at every step, moving lr to CUDA will be a bottleneck.
                if not isinstance(group["lr"], Tensor):
                    group["lr"] = torch.tensor(group["lr"], device=p.device)

                if "two_states" in state:
                    single_param_adam_4bit(
                        p.view(-1),
                        grad.view(-1),
                        state["step"],
                        state["two_states"],
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                    )
                else:
                    single_param_adam(
                        p.view(-1),
                        grad.view(-1),
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        None,
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                    )

        return loss


@torch.compile(fullgraph=True, dynamic=True)
def single_param_adam_4bit(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    two_states: Optim2State4bit,
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

    exp_avg, exp_avg_sq = two_states.dequantize()
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    two_states.copy_(exp_avg, exp_avg_sq)  # quantize

    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
    step_size = lr / bias_correction1
    p.addcdiv_(exp_avg, denom, value=-step_size)
