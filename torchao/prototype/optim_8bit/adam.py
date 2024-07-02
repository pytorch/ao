import torch
from torch import Tensor
from torch.optim import Optimizer

from .subclass import DTQ8bit


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
            group.setdefault("amsgrad", False)

    # follow bitsandbytes
    # only apply quantization for tensor with more than 4096 values
    # TODO: also skip 1D tensor? e.g. biases and norm scales
    def _new_zero_buffer(self, p: Tensor, signed: bool = True):
        if p.numel() >= 4096 and p.numel() % self.block_size == 0:
            out = DTQ8bit.zeros(p.shape, signed, self.block_size, device=p.device)
        else:
            out = torch.zeros_like(p)
        return out

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
                    state["exp_avg"] = self._new_zero_buffer(p.view(-1), signed=True)
                    state["exp_avg_sq"] = self._new_zero_buffer(p.view(-1), signed=False)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = self._new_zero_buffer(p.view(-1), signed=False)

                state["step"] += 1

                # flatten p and grad so that torch.compile won't recompile for tensors with different ndim
                # must explicitly convert lr to Tensor since torch.compile() will treat it as a constant
                # if it is a python float. practically, only lr is changed during training.
                # NOTE: if lr is change at every step, moving lr to CUDA will be a bottleneck.
                if not isinstance(group["lr"], Tensor):
                    group["lr"] = torch.tensor(group["lr"], device=p.device)
                single_param_adam_v1_(
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

                # with torch._fused_adam_()
                # single_param_adam_v2_(
                #     p.view(-1),
                #     grad.view(-1),
                #     state["step"],
                #     state["exp_avg"],
                #     state["exp_avg_sq"],
                #     state.get("max_exp_avg_sq", None),
                #     group["lr"],
                #     group["betas"][0],
                #     group["betas"][1],
                #     group["weight_decay"],
                #     group["eps"],
                # )

        return loss


# this will work with any optim state tensor subclass that implements aten.lerp.Scalar and aten.copy_.default
@torch.compile(fullgraph=True, dynamic=True)
def single_param_adam_v1_(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Tensor | None,
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


# torch._fused_adam_() cannot be compiled, thus we need to compile dequant and quant code separately.
def single_param_adam_v2_(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Tensor | None,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
):
    amsgrad = max_exp_avg_sq is not None

    # dequantize
    if isinstance(exp_avg, DTQ8bit):
        _dequantize = torch.compile(DTQ8bit.dequantize)
        exp_avg_fp32 = _dequantize(exp_avg)
        exp_avg_sq_fp32 = _dequantize(exp_avg_sq)
        max_exp_avg_sq_fp32 = _dequantize(max_exp_avg_sq) if amsgrad else None
    else:
        exp_avg_fp32 = exp_avg
        exp_avg_sq_fp32 = exp_avg_sq
        max_exp_avg_sq_fp32 = max_exp_avg_sq

    torch._fused_adam_(
        [p],
        [grad],
        [exp_avg_fp32],
        [exp_avg_sq_fp32],
        [max_exp_avg_sq_fp32] if amsgrad else [],
        [step],
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=False,
    )

    # quantize
    if isinstance(exp_avg, DTQ8bit):
        _quantize_copy_ = torch.compile(DTQ8bit.copy_)
        _quantize_copy_(exp_avg, exp_avg_fp32)
        _quantize_copy_(exp_avg_sq, exp_avg_sq_fp32)
        _quantize_copy_(max_exp_avg_sq, max_exp_avg_sq_fp32) if amsgrad else None
