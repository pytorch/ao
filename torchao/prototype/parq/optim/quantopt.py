# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from torchao import quantize_
from torchao.core.config import AOBaseConfig

from ..quant import Quantizer
from ..utils import HAS_DTENSOR, is_dtensor
from .proxmap import ProxMap

if HAS_DTENSOR:
    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.tensor.experimental import local_map
    from torch.distributed.tensor.placement_types import Shard


class QuantOptimizer(Optimizer):
    """QuantOptimizer assembles functionalities of the following objects:
    a base optimizer (e.g., SGD or AdamW)
        - update the latent variables for QAT
    a quantizer (e.g., UnifQuantizer, LSBQuantizer, LearnedScale)
        - update target quantization values for model parameters
    a proximal mapping (e.g, HardQuant/STE, PARQ, BinaryRelax)
        - update model parameters based on the above two updates
    Other parameters:
        - warmup_steps: int >= 0
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

    @property
    def state(self) -> defaultdict[Tensor, Any]:  # pyre-ignore[3]
        return self._state if hasattr(self, "_state") else self.base_optimizer.state

    @staticmethod
    def quantize_(
        p: Tensor,
        quants: Tensor,
        quantizer: Quantizer,
        b: int,
        dim: Optional[int] = None,
    ) -> Optional[Tensor]:
        """Optionally update the quantization targets `quants` in place.
        Return the quantized `p` as a by-product if `quant_update=True`.
        """
        q, Q = quantizer.quantize(p, b, dim=dim)  # pyre-ignore[28]
        quants.copy_(Q)
        return q

    @torch.no_grad()
    def torchao_quantize_(self, model: torch.nn.Module, config: AOBaseConfig):
        """Recursively call torchao.quantize_ on model using given config."""
        self.restore_latent_params()
        param_set = {
            p.data_ptr()
            for group in self.regularized_param_groups()
            for p in group["params"]
        }

        def inner_quantize_(model):
            for module in model.children():
                for param in module.parameters(recurse=False):
                    if param.data_ptr() in param_set:
                        quantize_(module, config)
                        break

                inner_quantize_(module)

        inner_quantize_(model)

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
        qat_state = state_dict.get("qat_state")
        # resume from check points usually not corresponds to saved num_steps
        # so allow explicit start_step computed from epochs * steps_per_epoc
        if start_step is not None:
            self.num_steps = start_step
        elif qat_state is not None:
            # hope discrepancy in num_steps does not cause major problem!
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

        if self.num_steps == self.warmup_steps:
            # first step of qat, save latent params, instead of restore
            self.save_latent_params()
        else:
            # qat: restore latent params for update by the base optimizer
            self.restore_latent_params()

        # call base optimizer step() method to update latent parameters
        loss = self.base_optimizer.step(closure=closure)  # pyre-ignore[6]

        if hasattr(self, "_state"):
            assert self.warmup_steps == 0
            # restore the temporary state to the base optimizer's state
            for p in self._state.keys():
                self.base_optimizer.state[p]["latent"] = self._state[p]["latent"]
            del self._state

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
            block_size = group.get("quant_block_size")
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

                # reshape p according to block size if specified
                if block_size is not None:
                    assert p.size(-1) % block_size == 0, (
                        f"{p.size(-1)=} is not divisible by {block_size=}"
                    )
                    assert p.dim() <= 2, f"Invalid {p.dim()=} for {block_size=}"
                    if p.dim() == 1:
                        p = p.unsqueeze(0)

                    # row-major ordering ensures this is correct
                    p = p.view(-1, block_size)

                # quantization by channel or by layer
                # update quantization targets periodically
                per_channel = self.quant_per_channel and p.dim() > 1
                if quant_update:
                    quant_size = self.quantizer.get_quant_size(b)

                    if per_channel:
                        quant_size = (p.size(0), quant_size)
                    state["quants"] = torch.empty(quant_size, device=p.device)
                    if is_dtensor(p):
                        state["quants"] = distribute_tensor(
                            state["quants"],
                            device_mesh=p.device_mesh,
                            placements=p.placements,
                        )

                dim = -1 if per_channel else None
                if per_channel and p.dim() > 2:
                    p = p.flatten(start_dim=1)

                q = None
                if quant_update:
                    qfunc = partial(
                        self.quantize_, quantizer=self.quantizer, b=b, dim=dim
                    )
                    if is_dtensor(p):
                        qfunc = local_map(
                            qfunc,
                            out_placements=[*p.placements],
                            in_placements=([Shard(0)], [Shard(0)]),
                        )
                    q = qfunc(p, state["quants"])

                # apply (step-dependent) proximal mapping in place
                pfunc = partial(
                    self.prox_map.apply_, step_count=self.num_steps, dim=dim
                )
                if is_dtensor(p):
                    pfunc = local_map(
                        pfunc,
                        out_placements=None,
                        in_placements=(
                            [Shard(0)],
                            None if q is None else [Shard(0)],
                            [Shard(0)],
                        ),
                    )
                inv_slope = pfunc(p, q, state["quants"])

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
        if self.warmup_steps == 0:
            assert len(self.state) == 0, "Expected empty state at first step()"
            # Maintain the invariant that `len(self.state) == 0` before first
            # self.base_optimizer.step() call by using a temporary state buffer
            self._state = defaultdict(dict)

        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "latent" not in state:
                        state["latent"] = p.detach().clone()
                    else:
                        state["latent"].copy_(p)
