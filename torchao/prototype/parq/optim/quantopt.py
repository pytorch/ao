# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any, Generator, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    PerGroup,
    PerRow,
    quantize_,
)
from torchao.quantization.quantize_.common import PackingFormat

from ..quant import Quantizer
from ..quant.quant_api import StretchedIntxWeightOnlyConfig
from ..quant.uniform_torchao import (
    _BIT_WIDTH_TO_DTYPE,
    Int4UnifTorchaoQuantizer,
    StretchedUnifTorchaoQuantizer,
    UnifTorchaoQuantizer,
)
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

    def regularized_param_groups(self) -> Generator[dict[str, Any], None, None]:
        """Yield parameter groups that need to be quantized."""
        for group in self.param_groups:
            if group.get("quant_bits", 16) < 16:
                yield group

    def _param_sets(self) -> Generator[set[int], None, None]:
        for group in self.regularized_param_groups():
            yield {p.data_ptr() for p in group["params"]}

    def get_filter_fns(
        self, module: nn.Module
    ) -> Generator[Callable[[nn.Module], bool], None, None]:
        def _filter_fn(module: nn.Module, *args, param_set) -> bool:
            for p in module.parameters(recurse=False):
                if p.data_ptr() in param_set:
                    return True
            return False

        for param_set in self._param_sets():
            yield partial(_filter_fn, param_set=param_set)

    def torchao_convert(
        self,
        model: nn.Module,
    ) -> None:
        """Converts model parameters to torchao quantized tensor subclasses."""
        model.eval()
        self.restore_latent_params()

        # TODO(lvj): find more robust way to identify embedding layers
        embed_data_ptrs = {
            module.weight.data_ptr()
            for module in model.modules()
            if isinstance(module, nn.Embedding)
        }

        for group, filter_fn in zip(
            self.regularized_param_groups(), self.get_filter_fns(model)
        ):
            quantizer = group.get("quantizer", self.quantizer)
            if not isinstance(quantizer, UnifTorchaoQuantizer):
                continue

            weight_dtype = _BIT_WIDTH_TO_DTYPE[group["quant_bits"]]
            granularity = (
                PerGroup(group["quant_block_size"])
                if "quant_block_size" in group
                else PerRow()
            )
            if isinstance(quantizer, Int4UnifTorchaoQuantizer):
                config = Int4WeightOnlyConfig(
                    group_size=group["quant_block_size"],
                    packing_format=PackingFormat.PLAIN,
                )
            elif isinstance(quantizer, StretchedUnifTorchaoQuantizer):
                config = StretchedIntxWeightOnlyConfig(
                    b=group["quant_bits"],
                    quant_min=quantizer.quant_min,
                    quant_max=quantizer.quant_max,
                    granularity=granularity,
                )
            elif all(p.data_ptr() in embed_data_ptrs for p in group["params"]):
                config = IntxWeightOnlyConfig(
                    weight_dtype=weight_dtype,
                    granularity=granularity,
                    mapping_type=quantizer.mapping_type,
                    packing_format=PackingFormat.UNPACKED_TO_INT8,
                    version=2,
                )
            else:
                config = Int8DynamicActivationIntxWeightConfig(
                    weight_dtype=weight_dtype,
                    weight_granularity=granularity,
                    weight_mapping_type=quantizer.mapping_type,
                    act_mapping_type=MappingType.ASYMMETRIC,
                    packing_format=PackingFormat.UNPACKED_TO_INT8,
                    version=2,
                )
            quantize_(model, config, filter_fn=filter_fn)

    @torch._disable_dynamo
    def state_dict(self) -> dict[str, Any]:
        return self.base_optimizer.state_dict()

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
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
            # Override quantizer if specified in the group
            quantizer = group.get("quantizer", self.quantizer)
            assert isinstance(quantizer, Quantizer), f"Invalid {quantizer=}"

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
                    quant_size = quantizer.get_quant_size(b)

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
                    qfunc = partial(self.quantize_, quantizer=quantizer, b=b, dim=dim)
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
    @torch.no_grad()
    def restore_latent_params(self) -> None:
        """Restore latent parameters as optimizer parameters"""
        for group in self.regularized_param_groups():
            for p in group["params"]:
                if p.requires_grad:
                    p.copy_(self.state[p]["latent"])

    @torch._disable_dynamo
    @torch.no_grad()
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
