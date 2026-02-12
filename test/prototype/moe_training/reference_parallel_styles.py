# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference implementations of expert parallelism copied from torchtitan
# to allow running tests without requiring torchtitan to be installed.
# See: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/expert_parallel.py

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle

from .reference_moe import _permute, _unpermute


class NoParallel(ParallelStyle):
    """Placeholder for no parallelization."""

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return module


class TensorParallel(ParallelStyle):
    """Tensor parallelism for MoE layers (matches torchtitan implementation)."""

    def _prepare_input_fn(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))
        return routed_input, num_tokens_per_expert

    def _partition_fn(self, name, module, device_mesh):
        from torch.distributed.tensor import distribute_tensor

        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )
        module.register_parameter(
            "w2", nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)]))
        )
        module.register_parameter(
            "w3", nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)]))
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torch.distributed.tensor import distribute_module

        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            self._prepare_input_fn,
        )


class ExpertParallel(ParallelStyle):
    """Expert parallelism for MoE layers (matches torchtitan implementation)."""

    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        from torch.distributed.tensor import distribute_tensor

        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.distributed._functional_collectives import (
            all_to_all_single,
            all_to_all_single_autograd,
        )

        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: torch.Tensor, device_mesh: DeviceMesh
    ) -> torch.Tensor:
        from torch.distributed._functional_collectives import all_to_all_single_autograd

        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torch.distributed.tensor import distribute_module

        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


class ExpertTensorParallel(ExpertParallel):
    """Combined expert and tensor parallelism for MoE layers (matches torchtitan implementation)."""

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs

        routed_input = DTensor.from_local(
            routed_input, device_mesh["tp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        inputs = (routed_input, num_tokens_per_expert)

        return super()._token_dispatch(mod, inputs, device_mesh["ep"])

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        from torch.distributed.tensor import distribute_tensor

        mod.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)])),
        )
        mod.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)])),
        )
        mod.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(mod.w3, device_mesh, [Shard(0), Shard(1)])),
        )

    def _token_combine(self, mod, routed_output, device_mesh):
        return super()._token_combine(mod, routed_output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torch.distributed.tensor import distribute_module

        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
