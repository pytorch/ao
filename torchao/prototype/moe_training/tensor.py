# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch._prims_common import suggest_memory_format
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.autograd.grad_mode import _unsafe_preserve_version_counter

from torchao.prototype.moe_training import _scaled_grouped_mm

logger: logging.Logger = logging.getLogger(__name__)


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


class ScaledGroupedMMTensor(torch.Tensor):
    """
    ScaledGroupedMMTensor is a simple tensor subclass that wraps a regular tensor
    and overrides the torch._grouped_mm op by dispatching to the
    differentiable _scaled_grouped_mm autograd function.
    """

    grouped_mm_func_names = {"_grouped_mm", "_grouped_mm.default"}
    offs_arg_name = "offs"

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
    ):
        self._data = tensor
        self._dtype = dtype

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # override the grouped mm op to use the differentiable _scaled_grouped_mm
        if func.__name__ in cls.grouped_mm_func_names:
            # Use torchao scaled grouped mm with dynamic quant for
            # "2d x 3d with offsets" case (used for routed experts).
            # Otherwise, fall back to regular grouped mm.
            #
            # TODO: support "3d x 3d without offsets" case, which is
            # used for shared experts. This is basically the grouped_mm
            # kernel handling a bmm.
            A, B = args[0], args[1]
            A_is_2d_or_3d = A.dim() in (2, 3)
            B_is_3d = B.dim() == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None
            logger.debug(f"A.shape={A.shape}, B.shape={B.shape}, has_offs={has_offs}")

            if A_is_2d_or_3d and B_is_3d:
                return _scaled_grouped_mm(
                    *args,
                    **kwargs,
                )

        # Disable torch_function by hand because we don't want
        # the wrapping behavior of the super() impl, go directly to dispatch
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # detach is special case
        if func == torch.ops.aten.detach.default:
            return ScaledGroupedMMTensor(args[0]._data, args[0]._dtype)

        # unwrap args and kwargs
        dtype: Optional[torch.dtype] = None

        def unwrap(t):
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._data

        args, kwargs = pytree.tree_map_only(
            ScaledGroupedMMTensor, unwrap, (args, kwargs or {})
        )

        # perform op
        out = func(*args, **kwargs)

        # return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # wrap outputs back into ScaledGroupedMMTensor for ops that do preserve subclass
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: ScaledGroupedMMTensor(x, dtype),
            out,
        )

    def __repr__(self):
        return f"ScaledGroupedMMTensor(data.dtype={self._data.dtype}, self.dtype={self._dtype})"

    def __tensor_flatten__(self):
        return ["_data"], {"_dtype": self._dtype}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return ScaledGroupedMMTensor(
            inner_tensors["_data"],
            flatten_spec["_dtype"],
        )

    # fsdp hooks based on https://github.com/pytorch/pytorch/blob/20e40492b046b9287726d3ec656117e4dc38f0e2/test/distributed/_composable/fsdp/test_fully_shard_extensions.py#L81
    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        all_gather_inputs = (self._data.to(mp_policy.param_dtype),)
        all_gather_metadata = ()
        logger.debug(f"fsdp_pre_all_gather: self._data.dtype={self._data.dtype}, param_dtype: {mp_policy.param_dtype}")
        return all_gather_inputs, all_gather_metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        logger.debug(f"fsdp_post_all_gather: data.dtype={data.dtype}, param_dtype: {param_dtype}")

        if out is not None:
            with torch.no_grad():
                out.copy_(data)
            return

        upcast_data = data.to(param_dtype)
        output = ScaledGroupedMMTensor(upcast_data, param_dtype)
        inner_tensors = (upcast_data,)
        return output, inner_tensors
