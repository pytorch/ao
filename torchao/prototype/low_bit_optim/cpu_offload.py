from typing import Type

import torch
from torch.optim.optimizer import Optimizer


class CPUOffloadOptimizer:
    def __init__(self, params, optimizer_class: Type[Optimizer], *, offload_gradients: bool = False, **kwargs) -> None:
        """Offload optimizer to CPU for single-GPU training. This will reduce GPU memory by the size of optimizer state.
        Optimizer step will be done on CPU.

        Args
            params: a list of parameters or parameter groups.
            optimizer_class: constructor of the base optimizer.
            offload_gradients: free GPU gradients once they are moved to CPU. Not compatible with gradient accumulation.
            kwargs: other keyword arguments to be passed to the base optimizer e.g. `lr`, `weight_decay`.

        NOTE: To minimize the amount of CPU<->GPU data transfer, we keep a copy of parameters and pre-allocate gradients
        memory on CPU. Therefore, expect your RAM usage to increase by 2x model size + optimizer state (which is 2x model
        size for Adam).

        NOTE: It is recommended not to `torch.compile()` your whole model when `CPUOffloadOptimizer` is used, as it
        prevents us from interleaving gradient device-to-host transfer with backward pass. To minimize such impact, you
        can compile parts of your model separately.
        """
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        self.param_cuda2cpu_map = dict()
        self.optim_dict = dict()
        self.stream = torch.cuda.Stream()

        # the queue maintains the order which param we should do optim step on first.
        self.queue = dict()

        def backward_hook(p_cuda):
            if p_cuda.grad is not None:
                p_cpu = self.param_cuda2cpu_map[p_cuda]

                # make sure backward for this param finishes
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    p_cpu.grad.copy_(p_cuda.grad, non_blocking=True)

                # we rely on CPython implementation of dictionary, which preserves insertion order.
                # if a param is added again (e.g. due to gradient accumulation), it is moved to the
                # end of the queue by removing and inserting it again.
                if p_cuda in self.queue:
                    del self.queue[p_cuda]
                self.queue[p_cuda] = self.stream.record_event()

                # this doesn't seem to work...
                if offload_gradients:
                    p_cuda.grad.record_stream(self.stream)
                    p_cuda.grad = None

        for param_group in param_groups:
            params = param_group.pop("params")

            for p_cuda in params:
                p_cpu = p_cuda.detach().cpu().pin_memory()
                p_cpu.grad = torch.empty_like(p_cpu, pin_memory=True)  # pre-allocate CPU grad
                self.param_cuda2cpu_map[p_cuda] = p_cpu

                p_cuda.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_cuda] = optimizer_class([{"params": p_cpu, **param_group}], **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for p_cuda, grad_d2h_event in self.queue.items():
            grad_d2h_event.synchronize()
            self.optim_dict[p_cuda].step()

            # submit more job to self.stream. it guarantees that we only start
            # moving param H2D once all backwards finish, since self.stream
            # will wait for current_stream when moving grad D2H.
            p_cpu = self.param_cuda2cpu_map[p_cuda]
            with torch.cuda.stream(self.stream):
                p_cuda.copy_(p_cpu, non_blocking=True)

        self.queue.clear()
        return loss

    def zero_grad(self, set_to_none=True):
        assert set_to_none

        # only clear CUDA grad. CPU grad will always be overwritten by CUDA grad.
        for p_cuda in self.param_cuda2cpu_map.keys():
            p_cuda.grad = None

    @property
    def param_groups(self):
        # each param group will only has 1 parameter
        # TODO: we might want to return the original param_groups instead.
        return sum((optim.param_groups for optim in self.optim_dict.values()), start=[])
