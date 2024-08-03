from typing import Type

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
                self.d2h_grad_stream.wait_stream(torch.cuda.current_stream())
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

        # we don't explicitly synchronize here, since forward will
        # wait for H2D transfer to finish.

    # redirect calls to base optimizer
    def __getattr__(self, name: str):
        return getattr(self.optim, name)


def _copy_grad_d2h(optim: Optimizer, param_cpu2cuda_map: dict, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        for param_group in optim.param_groups:
            for p_cpu in param_group["params"]:
                p_cuda = param_cpu2cuda_map[p_cpu]
                if p_cuda.grad is not None:
                    p_cpu.grad = p_cuda.grad.to("cpu", non_blocking=True)

                    # only deallocate p_cuda.grad once D2H copy finishes
                    p_cuda.grad.record_stream(stream)
                    p_cuda.grad = None


def _copy_param_h2d(optim: Optimizer, param_cpu2cuda_map: dict, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        for param_group in optim.param_groups:
            for p_cpu in param_group["params"]:
                if p_cpu.grad is not None:
                    p_cuda = param_cpu2cuda_map[p_cpu]
                    p_cuda.copy_(p_cpu, non_blocking=True)


class CPUOffloadOptimizerv2:
    def __init__(self, params, optimizer_class: Type[Optimizer], *, num_buckets: int = 10, **kwargs) -> None:
        if num_buckets < 2:
            raise ValueError(f"num_buckets should be >= 2. Received {num_buckets}")

        params = list(params)
        if isinstance(params[0], dict):
            raise ValueError("Param groups are currently not supported.")

        # copy param to CPU, so that optim state will be initialized on CPU
        # pin memory here, so that we can do fast async H2D transfer later
        self.param_cpu2cuda_map = dict()
        for i, p_cuda in enumerate(params):
            p_cpu = p_cuda.detach().cpu().pin_memory()
            params[i] = p_cpu
            self.param_cpu2cuda_map[p_cpu] = p_cuda

        buckets = [[] for _ in range(num_buckets)]
        bucket_sizes = [0] * num_buckets

        # sorted-greedy approach for multiway number partitioning
        # sort from largest to smallest
        params.sort(key=lambda x: x.numel(), reverse=True)
        for p in params:
            bin_idx = min(range(num_buckets), key=lambda x: bucket_sizes[x])
            buckets[bin_idx].append(p)
            bucket_sizes[bin_idx] += p.numel()

        self.optim_list = [optimizer_class(bucket, **kwargs) for bucket in buckets]
        self.streams = [torch.cuda.Stream() for _ in range(2)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # naive:
        # |--D2H transfer N grads--| |--CPU optim step N params--| |--H2D transfer N params--|
        #
        # interleave: transfer with CPU optim step
        # |--D2H transfer group 1--| |--CPU optim step group 1--| |-- H2D transfer group 1 --|
        #                            |-- D2H transfer group 2 --| |--CPU optim step group 2--| ...
        # therefore, we only need to wait for (D2H transfer group 1) and (H2D transfer group K)
        #
        # we will use 2 CUDA streams. optim step is always blocking on host.
        # host:             | optimizer 1 | | optimizer 2 | | optimizer 3 |
        # stream 1: |D2H 1|                 |H2D 1| |D2H 3|                 ...
        # stream 2:                 |D2H 2|                 |H2D 2| |D2H 4| ...

        # must wait until backward finishes
        torch.cuda.synchronize()

        # launch (D2H 1) and (D2H 2)
        _copy_grad_d2h(self.optim_list[0], self.param_cpu2cuda_map, self.streams[0])
        _copy_grad_d2h(self.optim_list[1], self.param_cpu2cuda_map, self.streams[1])

        for i, optim in enumerate(self.optim_list):
            stream = self.streams[i % 2]
            stream.synchronize()
            optim.step()
            _copy_param_h2d(optim, self.param_cpu2cuda_map, stream)

            # launch D2H kernel for the next next group
            if i < len(self.optim_list) - 2:
                _copy_grad_d2h(self.optim_list[i + 2], self.param_cpu2cuda_map, stream)

        stream.synchronize()
        return loss

    def zero_grad(self, set_to_none=True):
        for optim in self.optim_list:
            optim.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return sum((optim.param_groups for optim in self.optim_list), start=[])


class CPUOffloadOptimizerv3:
    def __init__(self, params, optimizer_class: Type[Optimizer], **kwargs) -> None:
        params = list(params)
        self.param_cuda2cpu_map = dict()
        self.optim_dict = dict()
        self.stream = torch.cuda.Stream()
        self.queue = []

        def backward_hook(p_cuda):
            if p_cuda.grad is not None:
                p_cpu = self.param_cuda2cpu_map[p_cuda]

                # make sure backward for this param finishes
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    p_cpu.grad.copy_(p_cuda.grad, non_blocking=True)

                grad_d2h_event = self.stream.record_event()
                self.queue.append((p_cuda, grad_d2h_event))

        for p_cuda in params:
            p_cpu = p_cuda.detach().cpu().pin_memory()
            p_cpu.grad = torch.empty_like(p_cpu, pin_memory=True)  # pre-allocate CPU grad
            self.param_cuda2cpu_map[p_cuda] = p_cpu

            p_cuda.register_post_accumulate_grad_hook(backward_hook)
            self.optim_dict[p_cuda] = optimizer_class([p_cpu], **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        for p_cuda, grad_d2h_event in self.queue:
            grad_d2h_event.synchronize()
            self.optim_dict[p_cuda].step()

            # submit more job to self.stream. it guarantees that we only start
            # moving param H2D once all backwards finish, since self.stream
            # will wait for current_stream when moving grad D2H.
            p_cpu = self.param_cuda2cpu_map[p_cuda]
            with torch.cuda.stream(self.stream):
                p_cuda.copy_(p_cpu, non_blocking=True)

        self.queue = []
        return

    def zero_grad(self, set_to_none=True):
        assert set_to_none

        # only clear CUDA grad. CPU grad will always be overwritten by CUDA grad.
        for p_cuda in self.param_cuda2cpu_map.keys():
            p_cuda.grad = None

    @property
    def param_groups(self):
        return sum((optim.param_groups for optim in self.optim_dict.values()), start=[])
