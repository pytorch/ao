# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import re
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch import nn

from .distributed import is_main_process

RE_PREFIX = ":"


def get_index_linspace(
    index_slope: float,
    n_indices: int,
    device: torch.device,
    max_val: Optional[float] = None,
):
    gamma_multiplier = (
        torch.linspace(1 - index_slope, 1 + index_slope, n_indices, device=device)
        .div_(2.0)
        .clamp_(min=0.0, max=max_val)
    )
    return gamma_multiplier


@contextmanager
def use_deterministic_algorithms():
    """Context manager to enable deterministic algorithms in PyTorch"""
    deterministic_restore = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic_restore)


class FuncDescriptor:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(instance)


def instantiate_module(module_name: str):
    prefix, name = module_name.rsplit(".", 1)
    module = getattr(import_module(prefix), name)
    return module


def get_param_groups(
    model: nn.Module,
    prune_config: Dict[Tuple[nn.Module, str], Any],
    skip_wd_names: Optional[Set[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    # Create list of regex patterns for matching parameter names
    re_pats = [
        re.compile(k[len(RE_PREFIX) :])
        for k in prune_config.keys()
        if isinstance(k, str) and k.startswith(RE_PREFIX)
    ]

    param_dict = {}
    seen_tensors = set()
    for param_name, param in model.named_parameters():
        module_name, _, param_basename = param_name.rpartition(".")
        parent_module = model.get_submodule(module_name) if module_name else model
        if param in seen_tensors:
            continue
        seen_tensors.add(param)

        group_key, group_val = None, None
        for re_pat in re_pats:
            if re_pat.match(param_name):
                group_key = re_pat.pattern
                group_val = prune_config[f"{RE_PREFIX}{group_key}"]
                break

        # Check for exact parameter or module name matches
        if group_key is None:
            module_cls = parent_module.__class__
            if param_name in prune_config:
                group_key = param_name
            elif (module_cls, param_basename) in prune_config:
                group_key = (module_cls, param_basename)
            elif (
                param_basename == "bias"
                or skip_wd_names
                and param_basename in skip_wd_names
            ):
                group_key, group_val = "no_wd", {"weight_decay": 0}
            else:
                group_key, group_val = "wd", {}

            if group_val is None:
                group_val = prune_config[group_key]

        param_dict.setdefault(group_key, group_val).setdefault("params", []).append(
            param
        )

    param_groups = list(param_dict.values())

    n_found_params = sum(len(v["params"]) for v in param_groups)
    n_expect_params = len(list(model.parameters()))
    assert n_found_params == n_expect_params, f"{n_found_params=}, {n_expect_params=}"

    if verbose and is_main_process():
        for k, v in param_dict.items():
            print(f"{k}: {len(v['params'])} params")
    return param_groups


def latent_svd(self, name=""):
    """Used when monkey patching the parameter to use SVD."""
    U = getattr(self, f"{name}_U")
    S = getattr(self, f"{name}_S")
    Vh = getattr(self, f"{name}_Vh")
    orig_shape = torch.Size(getattr(self, f"{name}_orig_shape"))
    return torch.linalg.multi_dot([U, torch.diag(S), Vh]).view(orig_shape)


def insert_svd_modules_(model: nn.Module, optimizer: torch.optim.Optimizer):
    """Replaces the parameters of the model with their SVD decompositions."""
    param_set = {
        p.data_ptr()
        for group in optimizer.regularized_param_groups()
        for p in group["params"]
        if group["group_type"] == "SVDGrouper"
    }

    def insert_inner_(model):
        for mn, module in model.named_children():
            params_to_add = {}
            for pn, p in module.named_parameters(recurse=False):
                if p.data_ptr() not in param_set:
                    continue

                k = int(optimizer.state[p]["sv_count"].item())
                assert k > 0, f"Invalid sv_count={k}"
                with instantiate_module("pat.group.SVDGrouper")(p) as grouper:
                    # patch parameter with SVD
                    module.register_buffer(
                        f"{pn}_orig_shape", torch.tensor(grouper.orig_shape)
                    )
                    U, S, Vh = grouper.U[:, :k], grouper.p[:k], grouper.Vh[:k]
                    for name, value in zip(
                        (f"{pn}_U", f"{pn}_S", f"{pn}_Vh"),
                        (U, S, Vh),
                    ):
                        params_to_add[name] = value

                    if is_main_process():
                        print(
                            f"{tuple(grouper.orig_shape)} -> "
                            f"{tuple(U.shape)}, {tuple(S.shape)}, {tuple(Vh.shape)}"
                        )

                module.__dict__.pop(pn, None)  # delete the original parameter
                setattr(
                    module.__class__,
                    pn,
                    FuncDescriptor(partial(latent_svd, name=pn)),
                )

            for name, value in params_to_add.items():
                module.register_parameter(
                    name, nn.Parameter(value, requires_grad=False)
                )
            del params_to_add

            insert_inner_(module)

    insert_inner_(model)
