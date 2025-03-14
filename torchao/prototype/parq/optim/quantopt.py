# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Callable
from functools import partial
from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from ..quant import LSBQuantizer, Quantizer
from .proxmap import ProxMap

try:
    from torch.distributed.tensor import DTensor

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False


class QuantOptimizer(Optimizer):
    """QuantOptimizer assembles functionalities of the following objects:
    a base optimizer (e.g., SGD or AdamW)
        - update the latent variables for QAT
    a quantizer (e.g., UnifQuantizer, LSBQuantizer, LearnedScale)
        - update target quantization values for model parameters
    a proximal mapping (e.g, HardQuant/STE, PARQ, BinaryRelax)
        - update model parameters based on the above two updates
    Other parameters:
        - warmup_steps: int > 0
        - quant_period: int > 0
        - quant_per_channel: True or False
        - quant_shrink: True or False
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        quantizer: Quantizer,
        prox_map: ProxMap,
        warmup_steps: int = 0,
        quant_period: int = 10,
        quant_per_channel: bool = False,
        quant_shrink: bool = False,
        anneal_wd_frac: float = 0.0,
    ) -> None:
        if not 0 <= anneal_wd_frac <= 1:
            raise ValueError(f"Invalid {anneal_wd_frac=} outside range [0.0, 1.0]")

        # need to reconstruct these objects if loading checkpoint
        self.base_optimizer = base_optimizer
        self.quantizer = quantizer
        self.prox_map = prox_map

        # need to store these attributes in state_dict for checkpoint
        self.warmup_steps = warmup_steps
        self.quant_period = quant_period
        self.quant_per_channel = quant_per_channel
        self.quant_shrink = quant_shrink
        self.anneal_wd_frac = anneal_wd_frac
        self.num_steps = 0

        # Initialize "cumu_lr" and latent params in optimizer states
        for group in self.regularized_param_groups():
            group["cumu_lr"] = 0.0
            if self.anneal_wd_frac > 0:
                group["initial_wd"] = group["weight_decay"]
        # NOTE: Filling state dict here cause Adam(W) error, which assumes
        # empty state[p] at first step() where optimizer states are initialized

    def __getattribute__(self, name: str):
        try:
            attr = super(Optimizer, self).__getattribute__(name)
        except AttributeError:
            attr = self.base_optimizer.__getattribute__(name)
        return attr

    def __repr__(self) -> str:
        base_optimizer = "\n    ".join(self.base_optimizer.__repr__().split("\n"))
        quantizer = self.quantizer.__class__.__name__
        prox_map = self.prox_map.__class__.__name__
        extra_repr = "\n  ".join(("(", base_optimizer, f"{quantizer=}", f"{prox_map=}"))
        return f"{self.__class__.__name__} {extra_repr}\n)"

    @staticmethod
    def quantize_(
        p: Tensor,
        quants: Tensor,
        quantizer: Quantizer,
        b: int,
        quant_update: bool,
        dim: Optional[int] = None,
    ) -> Optional[Tensor]:
        """Optionally update the quantization targets `quants` in place.
        Return the quantized `p` as a by-product if `quant_update=True`.
        """
        if quant_update:  # update Q for each channel
            q, Q = quantizer.quantize(p, b, dim=dim)  # pyre-ignore[28]
            quants.copy_(Q)
        else:
            q = None
        return q

    def regularized_param_groups(self):  # pyre-ignore[3]
        """Yield parameter groups that need to be quantized."""
        for group in self.param_groups:
            if group.get("quant_bits", 16) < 16:
                yield group

    @torch._disable_dynamo
    def state_dict(self) -> dict[str, Any]:
        state_dict = self.base_optimizer.state_dict()
        state_dict["qat_state"] = {"num_steps": self.num_steps}
        # quantizer and prox_map may also need to save states, can add here
        return state_dict

    @torch._disable_dynamo
    def load_state_dict(
        self, state_dict: dict[str, Any], start_step: Optional[int] = None
    ) -> None:
        qat_state = state_dict.pop("qat_state")
        # resume from check points usually not corresponds to saved num_steps
        # so allow explicit start_step computed from epochs * steps_per_epoc
        if start_step is not None:
            self.num_steps = start_step
        else:  # hope discrepancy in num_steps does not cause major problem!
            self.num_steps = qat_state["num_steps"]
        self.base_optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.num_steps < self.warmup_steps:
            # warmup stage: running the base optimizer only
            loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]
            self.num_steps += 1
            return loss

        # call base optimizer step() method to update latent parameters
        loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]

        if self.num_steps == self.warmup_steps:
            # first step of qat, save latent params, instead of restore
            self.save_latent_params()
        else:
            # qat: restore latent params for update by the base optimizer
            self.restore_latent_params()

        # check if it is time to update set of quantization values Q
        if (self.num_steps - self.warmup_steps) % self.quant_period == 0:
            quant_update = True
        else:
            quant_update = False

        for group in self.regularized_param_groups():
            # AProx in practice: ensure shrinkage coefficient >= 1
            group["cumu_lr"] += group["lr"]
            gamma = max(1.0, group["cumu_lr"])
            b = group["quant_bits"]
            inv_slope = 0.0
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                # save latent parameters, need detach()? or copy p)
                state["latent"].copy_(p)

                # in-place scaling of parameters by 1/gamma if specified
                if self.quant_shrink:
                    p.div_(gamma)

                # quantization by channel or by layer
                # update quantization targets periodically
                per_channel = self.quant_per_channel and p.dim() > 1
                if quant_update:
                    quants_size = 3 if b == 0 else 2**b
                    if per_channel:
                        quants_size = (p.size(0), quants_size)
                    state["quants"] = torch.empty(
                        quants_size, device=p.device
                    )  # pyre-ignore[6]

                # avoid type mismatch between sharded and full tensors
                if HAS_DTENSOR and isinstance(p, DTensor):
                    p = p.full_tensor()

                dim = -1 if per_channel else None
                if per_channel and p.dim() > 2:
                    p = p.flatten(start_dim=1)

                # NOTE: for LSBQ and optimal=False, use faster per-channel
                # implementation instead of vmap
                if isinstance(self.quantizer, LSBQuantizer) and self.quantizer.optimal:
                    qfunc = partial(
                        self.quantize_,
                        quantizer=self.quantizer,
                        b=b,
                        quant_update=quant_update,
                    )
                    q = torch.vmap(qfunc, in_dims=0, out_dims=0)(p, state["quants"])
                else:
                    q = self.quantize_(
                        p, state["quants"], self.quantizer, b, quant_update, dim=dim
                    )

                # apply (step-dependent) proximal mapping in place
                inv_slope = self.prox_map.apply_(  # pyre-ignore[28]
                    p, q, state["quants"], self.num_steps, dim=dim
                )

            # quantized parameters share the same PARQ inverse slope
            if inv_slope:
                if self.anneal_wd_frac > 0:
                    group["weight_decay"] = (
                        inv_slope * self.anneal_wd_frac * group["initial_wd"]
                        + (1 - self.anneal_wd_frac) * group["initial_wd"]
                    )
                group["inv_slope"] = inv_slope  # save for tensorboard

        self.num_steps += 1
        return loss

    @torch._disable_dynamo
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    p.copy_(self.state[p]["latent"])

    @torch._disable_dynamo
    def save_latent_params(self) -> None:
        """Save updated latent parameters before applying prox-map"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "latent" not in state:
                        state["latent"] = p.detach().clone()
                    else:
                        state["latent"].copy_(p)
