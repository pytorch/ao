# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Type

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from torchao.utils import get_available_devices


# NOTE: We make this inherit Optimizer so it works with PyTorch's built-in LR
# schedulers. (those schedulers specifically check for instances of Optimizer).
# However, it won't behave exactly like Optimizer e.g. we don't call
# Optimizer.__init__(), there is no self.defaults.
class CPUOffloadOptimizer(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        *,
        offload_gradients: bool = False,
        minimal_size: int = 4096,
        **kwargs,
    ) -> None:
        """Offload optimizer to CPU for single-GPU training. This will reduce GPU memory by the size of optimizer state.
        Optimizer step will be done on CPU.

        Args
            params: a list of parameters or parameter groups.
            optimizer_class: constructor of the base optimizer. Defaults to :class:`torch.optim.AdamW`.
            offload_gradients: free GPU gradients once they are moved to CPU. Not compatible with gradient accumulation.
            minimal_size: tensors smaller than this are kept on the GPU, to avoid excessively many small transfers.
            kwargs: other keyword arguments to be passed to the base optimizer e.g. `lr`, `weight_decay`.
        """
        # default to fused CPU AdamW
        if optimizer_class is torch.optim.AdamW and "fused" not in kwargs:
            kwargs.update(fused=True)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        # any parameter smaller than minimal size will be handled by the on-device optimizer d_opt
        self.minimal_size = minimal_size
        self.d_opt = None
        self.d_param_groups = []

        self.param_d2h_map = dict()
        self.optim_dict = dict()
        self.device = get_available_devices()[-1]
        assert self.device in [
            "cuda",
            "xpu",
        ], "CPU Offload currently only supports CUDA & XPU"
        self.stream = getattr(torch, self.device).Stream()

        # the queue maintains the order which param we should do optim step on first.
        self.queue = dict()

        def backward_hook(p_device):
            if p_device.grad is not None:
                p_host = self.param_d2h_map[p_device]

                # make sure backward for this param finishes
                self.stream.wait_stream(getattr(torch, self.device).current_stream())
                with getattr(torch, self.device).stream(self.stream):
                    p_host.grad.copy_(p_device.grad, non_blocking=True)

                # we rely on CPython implementation of dictionary, which preserves insertion order.
                # if a param is added again (e.g. due to gradient accumulation), it is moved to the
                # end of the queue by removing and inserting it again.
                if p_device in self.queue:
                    del self.queue[p_device]
                self.queue[p_device] = self.stream.record_event()

                # deallocate DEVICE gradients once D2H transfer finishes.
                if offload_gradients:
                    p_device.grad.record_stream(self.stream)
                    p_device.grad = None

        for param_group in param_groups:
            params = param_group.pop("params")
            retained_params = []

            for p_device in params:
                if not p_device.requires_grad:
                    continue

                if p_device.numel() < self.minimal_size:
                    retained_params.append(p_device)
                    continue

                # pre-allocate CPU params and grads
                p_host = torch.empty_like(p_device, device="cpu", pin_memory=True)
                p_host.grad = torch.empty_like(p_host, pin_memory=True)

                p_host.copy_(p_device.detach(), non_blocking=True)
                self.param_d2h_map[p_device] = p_host

                p_device.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[p_device] = optimizer_class(
                    [{"params": p_host, **param_group}], **kwargs
                )

            if len(retained_params) > 0:
                self.d_param_groups.append({"params": retained_params, **param_group})

        if len(self.d_param_groups) > 0:
            self.d_opt = optimizer_class(self.d_param_groups, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # handle small parameters on the GPU, in parallel with the CPU calls below
        if self.d_opt is not None:
            self.d_opt.step()

        for p_device, grad_d2h_event in self.queue.items():
            grad_d2h_event.synchronize()
            self.optim_dict[p_device].step()

            # submit more job to self.stream. it guarantees that we only start
            # moving param H2D once all backwards finish, since self.stream
            # will wait for current_stream when moving grad D2H.
            p_host = self.param_d2h_map[p_device]
            with getattr(torch, self.device).stream(self.stream):
                p_device.copy_(p_host, non_blocking=True)

        # make sure param H2D finishes before the next forward pass
        self.stream.synchronize()
        self.queue.clear()
        return loss

    def zero_grad(self, set_to_none=True):
        assert set_to_none

        # only clear DEVICE grad. CPU grad will always be overwritten by DEVICE grad.
        for p_device in self.param_d2h_map.keys():
            p_device.grad = None

        if self.d_opt is not None:
            self.d_opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        # each param group will only has 1 parameter
        # TODO: we might want to return the original param_groups instead.
        return sum(
            (optim.param_groups for optim in self.optim_dict.values()),
            start=self.d_param_groups,
        )

    def state_dict(self):
        state_dict = {
            "offloaded": [optim.state_dict() for optim in self.optim_dict.values()]
        }
        if self.d_opt:
            state_dict["on-device"] = self.d_opt.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for optim, optim_state_dict in zip(
            self.optim_dict.values(), state_dict["offloaded"]
        ):
            optim.load_state_dict(optim_state_dict)

        if self.d_opt:
            self.d_opt.load_state_dict(state_dict["on-device"])
        elif "on-device" in state_dict:
            raise ValueError(
                "loaded state dict has a 'on-device' parameter group not present in the optimizer"
            )
