import torch
from torch.optim.optimizer import Optimizer


class CPUOffloadOptimizer:
    def __init__(self, base_optimizer: Optimizer) -> None:
        self.optim = base_optimizer
        self.param_cuda2cpu_map = dict()
        self.d2h_grad_stream = torch.cuda.Stream()

        def copy_grad_hook(p_cuda):
            if p_cuda.grad is not None:
                p_cpu = self.param_cuda2cpu_map[p_cuda]
                with torch.cuda.stream(self.d2h_grad_stream):
                    p_cpu.grad = p_cuda.grad.to("cpu", non_blocking=True)

                # only deallocate p_cuda.grad once D2H copy finishes
                p_cuda.grad.record_stream(self.d2h_grad_stream)
                p_cuda.grad = None

        # swap param in param_groups with CPU param
        for param_group in base_optimizer.param_groups:
            for i, p in enumerate(param_group["params"]):
                p_cpu = p.detach().cpu().pin_memory()
                param_group["params"][i] = p_cpu
                self.param_cuda2cpu_map[p] = p_cpu
                p.register_post_accumulate_grad_hook(copy_grad_hook)

    @torch.no_grad()
    def step(self, closure=None):
        # copy gradients from CUDA to CPU
        # for p_cuda, p_cpu in self.param_cuda2cpu_map.items():
        #     if p_cuda.grad is not None:
        #         p_cpu.grad = p_cuda.grad.to("cpu", non_blocking=True)
        #         p_cuda.grad = None
        # torch.cuda.synchronize()

        self.d2h_grad_stream.synchronize()
        self.optim.step(closure)

        # copy updated param from CPU to CUDA
        for p_cuda, p_cpu in self.param_cuda2cpu_map.items():
            p_cuda.copy_(p_cpu, non_blocking=True)

    # redirect calls to base optimizer
    def __getattr__(self, name: str):
        return getattr(self.optim, name)
