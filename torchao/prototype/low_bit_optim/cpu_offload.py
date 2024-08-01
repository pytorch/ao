import torch
from torch.optim.optimizer import Optimizer


class CPUOffloadOptimizer:
    def __init__(self, base_optimizer: Optimizer) -> None:
        self.optim = base_optimizer
        self.param_cpu2cuda_map = dict()

        # swap param in param_groups with CPU param
        for param_group in base_optimizer.param_groups:
            for i, p in enumerate(param_group["params"]):
                p_cpu = p.detach().cpu().pin_memory()
                param_group["params"][i] = p_cpu
                self.param_cpu2cuda_map[p_cpu] = p

    @torch.no_grad()
    def step(self, closure=None):
        # copy gradients from CUDA to CPU
        for p_cpu, p_cuda in self.param_cpu2cuda_map.items():
            if p_cuda.grad is not None:
                p_cpu.grad = p_cuda.grad.to("cpu", non_blocking=True)
                p_cuda.grad = None
        torch.cuda.synchronize()

        self.optim.step(closure)

        # copy updated param from CPU to CUDA
        for p_cpu, p_cuda in self.param_cpu2cuda_map.items():
            p_cuda.copy_(p_cpu, non_blocking=True)

    # redirect calls to base optimizer
    def __getattr__(self, name: str):
        return getattr(self.optim, name)
