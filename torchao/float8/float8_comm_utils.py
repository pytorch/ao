# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format
from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_delayed,
    hp_tensor_to_float8_dynamic,
)

from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
)

from torchao.float8.float8_utils import e4m3_dtype, EPS

from torchao.float8.fsdp_utils import _ops_to_preserve_subclass

class Float8CommWeightWithDynamicCastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
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
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return Float8CommWeightWithDynamicCastTensor(
                args[0]._tensor
            )

        def unwrap(t):
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            Float8CommWeightWithDynamicCastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, lambda x: Float8CommWeightWithDynamicCastTensor(x), out
        )

    def __tensor_flatten__(self):
        if self._precomputed_scale:
            return ["_tensor", "_precomputed_scale"], None
        else:
            return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return Float8CommWeightWithDynamicCastTensor(
            inner_tensors["_tensor"],
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return f"Float8CommWeightWithDynamicCastTensor(tensor={self._tensor})"

    def fsdp_pre_all_gather(self, mesh):
        # print("pre-all-gather", self._tensor.shape, self._tensor)
        if self._precomputed_scale is not None:
            float8_tensor = hp_tensor_and_scale_to_float8(
                self._tensor,
                self._precomputed_scale,
                torch.float8_e4m3fn,
                None, # linear_mm_config
                GemmInputRole.WEIGHT,
            )
        else:
            float8_tensor = hp_tensor_to_float8_dynamic(
                self._tensor,
                e4m3_dtype,
                None, # linear_mm_config
                reduce_amax=True,
                gemm_input_role=GemmInputRole.WEIGHT,
            )
        # print("float8_tensor", float8_tensor)
        return (float8_tensor._data,), (float8_tensor._scale,)

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
        orig_weight = Float8Tensor(
            data,
            scale,
            param_dtype,
            None, # linear_mm_config
            gemm_input_role=GemmInputRole.WEIGHT,
        ) 
        orig_weight = orig_weight.to_original_precision()
        if out is not None:
            from torch.distributed._tensor import DTensor
            assert isinstance(out, torch.Tensor) or (isinstance(out, DTensor) and isinstance(out._local_tensor, torch.Tensor))
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [3072, 768]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
            # out.copy_(orig_weight)
            return
        # print("post-all=gather", orig_weight.shape, orig_weight)
        return orig_weight, (data,)


class Float8CommWeightWithDelayedCastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        is_amax_initialized: bool,
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
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        is_amax_initialized: bool,
    ):
        self._tensor = tensor
        self._amax_buffer = amax_buffer
        self._amax_history_buffer = amax_history_buffer
        self._scale_buffer = scale_buffer

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = is_amax_initialized

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return Float8CommWeightWithDelayedCastTensor(
                args[0]._tensor,
                args[0]._amax_buffer,
                args[0]._amax_history_buffer,
                args[0]._scale_buffer,
                args[0].is_amax_initialized,
            )
        amax_buffer: Optional[torch.Tensor] = None
        amax_history_buffer: Optional[torch.Tensor] = None
        scale_buffer: Optional[torch.Tensor] = None
        is_amax_initialized: Optional[bool] = None

        def unwrap(t):
            nonlocal amax_buffer
            if amax_buffer is None:
                amax_buffer = t._amax_buffer
            nonlocal amax_history_buffer
            if amax_history_buffer is None:
                amax_history_buffer = t._amax_history_buffer
            nonlocal scale_buffer
            if scale_buffer is None:
                scale_buffer = t._scale_buffer
            nonlocal is_amax_initialized
            if is_amax_initialized is None:
                is_amax_initialized = t.is_amax_initialized
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            Float8CommWeightWithDelayedCastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: Float8CommWeightWithDelayedCastTensor(
                x,
                amax_buffer,
                amax_history_buffer,
                scale_buffer,
                is_amax_initialized,
            ),
            out,
        )

    def __tensor_flatten__(self):
        return (
            [
                "_tensor",
                "_amax_buffer",
                "_amax_history_buffer",
                "_scale_buffer",
            ],
            {
                "is_amax_initialized": self.is_amax_initialized,
            },
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return Float8CommWeightWithDelayedCastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_amax_buffer"],
            inner_tensors["_amax_history_buffer"],
            inner_tensors["_scale_buffer"],
            metadata["is_amax_initialized"],
        )

    def __repr__(self):
        return f"WeightWithDelayedFloat8CastTensor(tensor={self._tensor}, amax_buffer={self._amax_buffer}, scale_buffer={self._scale_buffer})"

    def fsdp_pre_all_gather(self, mesh):
        # initialize if needed
        # TODO(before land): ensure settings are consistent between Float8Linear and here
        if not self.is_amax_initialized:
            from torchao.float8.float8_linear import (
                _maybe_initialize_amaxes_scales_for_float8_cast,
            )

            _maybe_initialize_amaxes_scales_for_float8_cast(
                self._tensor,
                self._amax_buffer,
                self._amax_history_buffer,
                self._scale_buffer,
                "max",  # TODO(before land): read this from parent
                e4m3_dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.is_amax_initialized = True

        float8_tensor = hp_tensor_to_float8_delayed(
            self._tensor,
            self._scale_buffer,
            e4m3_dtype,
            self._amax_buffer,
            None, # linear_mm_config
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

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
            assert isinstance(out, torch.Tensor) or (isinstance(out, DTensor) and isinstance(out._local_tensor, torch.Tensor))
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            None, # linear_mm_config
            gemm_input_role=GemmInputRole.WEIGHT,
        ).to_original_precision(), (data,)


class Float8CommWeightWithStaticCastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        static_scale: torch.Tensor,
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
        static_scale: torch.Tensor,
    ):
        self._tensor = tensor
        self._static_scale = static_scale

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return Float8CommWeightWithStaticCastTensor(
                args[0]._tensor, args[0]._static_scale
            )
        static_scale: Optional[torch.Tensor] = None

        def unwrap(t):
            nonlocal static_scale
            if static_scale is None:
                static_scale = t._static_scale
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            Float8CommWeightWithStaticCastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor, 
            lambda x: Float8CommWeightWithStaticCastTensor(x, static_scale), 
            out,
        )

    def __tensor_flatten__(self):
        return ["_tensor", "_static_scale"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return Float8CommWeightWithStaticCastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_static_scale"],
        )

    def __repr__(self):
        return f"WeightWithStaticFloat8CastTensor(tensor={self._tensor}, static_scale={self._static_scale})"

    def fsdp_pre_all_gather(self, mesh):
        float8_tensor = hp_tensor_and_scale_to_float8(
            self._tensor,
            self._static_scale,
            torch.float8_e4m3fn,
            None, # linear_mm_config
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

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
            assert isinstance(out, torch.Tensor) or (isinstance(out, DTensor) and isinstance(out._local_tensor, torch.Tensor))
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            None, # linear_mm_config
            gemm_input_role=GemmInputRole.WEIGHT,
        ).to_original_precision(), (data,)
