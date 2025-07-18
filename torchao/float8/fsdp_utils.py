# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format

from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_training_tensor import (
    Float8TrainingTensor,
    GemmInputRole,
    LinearMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import EPS


@torch.no_grad()
def precompute_float8_dynamic_scale_for_fsdp(module: nn.Module) -> None:
    """
    Calculate scale dynamically for all float8 parameters.
    This should be run after the optimizer step. It performs a single all-reduce to compute the
    scales for all float8 weights.
    Example usage:
        model(input).sum().backward()
        optim.step()
        precompute_float8_dynamic_scale_for_fsdp(model)
    """
    from torch.distributed._tensor import DTensor

    from torchao.float8.float8_linear import Float8Linear

    float8_linears: List[Float8Linear] = [
        m
        for m in module.modules()
        if isinstance(m, Float8Linear)
        and isinstance(m.weight, DTensor)
        and isinstance(m.weight._local_tensor, WeightWithDynamicFloat8CastTensor)
    ]
    weights: List[DTensor] = [float8_linear.weight for float8_linear in float8_linears]
    target_dtypes: Set[torch.dtype] = {
        float8_linear.config.cast_config_weight.target_dtype
        for float8_linear in float8_linears
    }

    if not weights:
        return
    (target_dtype,) = target_dtypes

    # inf-norm is equivalent to max(abs(w))
    max_weights = torch._foreach_norm(weights, ord=math.inf)  # Partial
    amax_tensor = torch.stack(max_weights)  # Partial
    # clamp is dispatched through DTensor
    # it will issue a single all-reduce
    amax_tensor = torch.clamp(amax_tensor, EPS)  # Replicate
    # keep consistent with float8_utils.amax_to_scale
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    origin_dtype = amax_tensor.dtype
    amax_tensor = amax_tensor.to(torch.float64)
    scale_tensor = torch.finfo(target_dtype).max / amax_tensor  # Replicate
    if origin_dtype is torch.float16:
        scale_tensor = torch.clamp(scale_tensor, max=torch.finfo(torch.float16).max)
    local_scale_tensor = scale_tensor.to_local().to(torch.float32)
    for i, float8_linear in enumerate(float8_linears):
        float8_linear.weight._local_tensor._precomputed_scale = local_scale_tensor[i]


# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}

# How Tensor Parallel (TP) and FSDP2 work

# Initialization: apply TP first then FSDP2
# nn.Linear(weight=torch.Tensor)
#      |
#      | apply float8 linear, `convert_to_float8_training`
#      |
# Float8Linear(weight=WeightWithDynamicFloat8CastTensor)
#      |
#      | apply tensor parallel, `parallelize_module` shards rowwise/colwise
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([0, 1], mesh_dim_names=('tp',)),
#                             placements=(Shard(dim=0),)))
#      |
#      | apply FSDP2, `fully_shard` shards rowwise (dim=0)
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp')),
#                             placements=(Shard(dim=0), Shard(dim=0))))

# Forward and backward: FSDP runs first then TP
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp')),
#                             placements=(Shard(dim=0), Shard(dim=0))))
#      |
#      |   FSDP unshards parameters within dp mesh
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([0, 1], mesh_dim_names=('tp',)),
#                             placements=(Shard(dim=0),)))
#      |
#      |   TP compute with torch.mm(input, weight)


class WeightWithDynamicFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicFloat8CastTensor(
                args[0]._tensor, args[0]._linear_mm_config, args[0]._dtype
            )
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDynamicFloat8CastTensor(x, mm_config, dtype),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        if self._precomputed_scale:
            tensors.append("_precomputed_scale")
        return tensors, {"mm_config": self._linear_mm_config, "dtype": self._dtype}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicFloat8CastTensor(
            inner_tensors["_tensor"],
            flatten_spec["mm_config"],
            flatten_spec["dtype"],
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return f"WeightWithDynamicFloat8CastTensor(tensor={self._tensor}, linear_mm_config={self._linear_mm_config}, dtype={self._dtype})"

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            float8_training_tensor = hp_tensor_and_scale_to_float8(
                self._tensor,
                self._precomputed_scale,
                self._dtype,
                self._linear_mm_config,
                GemmInputRole.WEIGHT,
            )
        else:
            float8_training_tensor = hp_tensor_to_float8_dynamic(
                self._tensor,
                self._dtype,
                self._linear_mm_config,
                reduce_amax=True,
                gemm_input_role=GemmInputRole.WEIGHT,
                device_mesh=mesh,
            )
        return (float8_training_tensor._data,), (float8_training_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8TrainingTensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8TrainingTensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8TrainingTensor or DTensor(_local_tensor=Float8TrainingTensor), but got {out}"
                )
            return
        return Float8TrainingTensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)


# Needed to allowlist this subclass for deserialization used for restoring checkpoints.
torch.serialization.add_safe_globals([WeightWithDynamicFloat8CastTensor])
