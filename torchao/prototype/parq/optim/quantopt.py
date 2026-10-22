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

from torchao.quantization.quant_api import _is_linear, quantize_

from ..quant import Quantizer, UnifTorchaoQuantizer
from ..quant.config_torchao import (
    _attach_hf_quantization_config,
    _get_config_from_quantizer,
    _is_hf_model,
)
from ..utils import HAS_DTENSOR, is_dtensor
from .proxmap import ProxMap

if HAS_DTENSOR:
    from torch.distributed.tensor import empty as dtensor_empty
    from torch.distributed.tensor.experimental import local_map


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
        group_quantizer_map: Optional[dict[int, Quantizer]] = None,
    ) -> None:
        if not 0 <= anneal_wd_frac <= 1:
            raise ValueError(f"Invalid {anneal_wd_frac=} outside range [0.0, 1.0]")

        # need to reconstruct these objects if loading checkpoint
        self.base_optimizer = base_optimizer
        self.quantizer = quantizer
        self.group_quantizer_map = group_quantizer_map
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

    @property
    def num_steps(self) -> int:
        for group in self.regularized_param_groups():
            return group.setdefault("num_steps", 0)

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        for group in self.regularized_param_groups():
            group["num_steps"] = value
            return

    @num_steps.deleter
    def num_steps(self) -> None:
        for group in self.regularized_param_groups():
            group.pop("num_steps", None)
            return

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

    def _get_quantizer(self, group_idx: int) -> Optional[Quantizer]:
        if self.group_quantizer_map and group_idx in self.group_quantizer_map:
            return self.group_quantizer_map[group_idx]
        return self.quantizer

    def torchao_convert(
        self,
        model: nn.Module,
        weight_only: bool = False,
        embed_weight_only: bool = False,
    ) -> None:
        """Converts model parameters to torchao quantized tensor subclasses."""
        model.eval()
        self.restore_latent_params()

        # TODO(lvj): find more robust way to identify embedding layers
        embed_data_ptrs = set()
        linear_data_ptrs = set()
        embed_modules = []
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embed_modules.append(module)
                embed_data_ptrs.add(module.weight.data_ptr())
            elif _is_linear(module) and module.weight.data_ptr() not in embed_data_ptrs:
                linear_data_ptrs.add(module.weight.data_ptr())

        tied_embeddings = False
        if not embed_weight_only and getattr(model, "_tied_weights_keys", None):
            # Workaround for dynamic activations on tied embeddings
            tied_embeddings = True
            for module in embed_modules:
                setattr(module, "bias", None)

        filter_fns = []
        configs = []
        attach_hf_config = _is_hf_model(model)
        all_linear_layers_idx = -1
        for i, (group, filter_fn) in enumerate(
            zip(self.regularized_param_groups(), self.get_filter_fns(model))
        ):
            filter_fns.append(filter_fn)
            quantizer = self._get_quantizer(i)
            if not isinstance(quantizer, UnifTorchaoQuantizer) or not group["params"]:
                configs.append(None)
                continue

            if set((p.data_ptr() for p in group["params"])) == linear_data_ptrs:
                all_linear_layers_idx = i

            device = group["params"][0].device
            any_embed = any(p.data_ptr() in embed_data_ptrs for p in group["params"])
            config = _get_config_from_quantizer(
                quantizer,
                weight_only or (any_embed and not tied_embeddings),
                device,
                group["quant_bits"],
                group.get("quant_block_size"),
            )
            configs.append(config)

        filter_fns_orig = filter_fns[:]
        configs_orig = configs[:]

        # If one group has all the linear layers, then set its config as default
        if all_linear_layers_idx > -1:
            module_to_config = {"_default": configs[all_linear_layers_idx]}
            del filter_fns[all_linear_layers_idx]
            del configs[all_linear_layers_idx]
        else:
            module_to_config = None

        if attach_hf_config:
            _attach_hf_quantization_config(model, filter_fns, configs, module_to_config)

        for config, filter_fn in zip(configs_orig, filter_fns_orig):
            quantize_(model, config, filter_fn=filter_fn)

    @torch._disable_dynamo
    def state_dict(self) -> dict[str, Any]:
        state_dict = self.base_optimizer.state_dict()
        if self.quant_period == 1:
            # Targets are rebuilt on the very next update. Keep the transient
            # local cache out of checkpoints and parameter-shaped optimizer IO.
            state_dict["state"] = {
                key: {name: value for name, value in state.items() if name != "quants"}
                if isinstance(state, dict)
                else state
                for key, state in state_dict["state"].items()
            }
        return state_dict

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

        for i, group in enumerate(self.regularized_param_groups()):
            # Override quantizer if specified in the group
            quantizer = self._get_quantizer(i)

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

                # Complete per-row/input-column groups are independent across
                # shards. Mutate the original local storage directly, following
                # PAT's rule that in-place prox updates must reach their owner.
                local_dtensor = None
                local_parameter = None
                if is_dtensor(p) and self.quant_per_channel and p.dim() == 2:
                    if p.device_mesh.get_coordinate() is None:
                        continue
                    if block_size is None and any(
                        placement.is_shard() and placement.dim == 1
                        for placement in p.placements
                    ):
                        raise ValueError(
                            "A per-row quantization group crosses an input-column shard"
                        )
                    local_dtensor = p
                    local_parameter = p.to_local()
                    p = local_parameter.contiguous()

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
                    quant_levels = quantizer.get_quant_size(b)
                    quant_size = (
                        (p.size(0), quant_levels) if per_channel else (quant_levels,)
                    )
                    quant_owner = local_dtensor if local_dtensor is not None else p
                    if local_dtensor is not None:
                        # Keep the original row/column-group axes in distributed
                        # state. Flattening global uneven row shards would assign
                        # different numbers of groups than the owning parameters.
                        quant_size = (
                            (
                                local_dtensor.size(0),
                                local_dtensor.size(1) // block_size,
                                quant_levels,
                            )
                            if block_size is not None
                            else (local_dtensor.size(0), quant_levels)
                        )
                    quants = state.get("quants")
                    if (
                        quants is None
                        or tuple(quants.shape) != quant_size
                        or quants.device != p.device
                    ):
                        # Allocate only owned state; there is no data to scatter.
                        # The next quantize_ call overwrites every local target.
                        state["quants"] = (
                            dtensor_empty(
                                quant_size,
                                device_mesh=quant_owner.device_mesh,
                                placements=quant_owner.placements,
                            )
                            if is_dtensor(quant_owner)
                            else torch.empty(quant_size, device=p.device)
                        )

                quants = state["quants"]
                if local_dtensor is not None:
                    quants = quants.to_local().reshape(-1, quants.size(-1))

                dim = -1 if per_channel else None
                if per_channel and p.dim() > 2:
                    p = p.flatten(start_dim=1)

                q = None
                if p.numel() == 0:
                    # Empty owning shards still carry canonical distributed
                    # state, but affine primitives cannot infer their qparams.
                    q = p
                elif quant_update:
                    qfunc = partial(self.quantize_, quantizer=quantizer, b=b, dim=dim)
                    if is_dtensor(p):
                        qfunc = local_map(
                            qfunc,
                            out_placements=[*p.placements],
                            in_placements=(p.placements, quants.placements),
                            redistribute_inputs=False,
                        )
                    q = qfunc(p, quants)

                # apply (step-dependent) proximal mapping in place
                pfunc = partial(
                    self.prox_map.apply_, step_count=self.num_steps, dim=dim
                )
                if is_dtensor(p):
                    pfunc = local_map(
                        pfunc,
                        out_placements=None,
                        in_placements=(
                            p.placements,
                            None if q is None else q.placements,
                            quants.placements,
                        ),
                        redistribute_inputs=False,
                    )
                inv_slope = pfunc(p, q, quants)
                if local_parameter is not None and not local_parameter.is_contiguous():
                    # Column-shard views may have gaps between rows. Group on
                    # contiguous workspace, then copy the projection to its owner.
                    local_parameter.copy_(p.reshape_as(local_parameter))

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
