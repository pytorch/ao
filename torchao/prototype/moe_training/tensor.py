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
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from torchao.prototype.moe_training import _scaled_grouped_mm
from torchao.prototype.moe_training.conversion_utils import MoEScalingType

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
    torch.ops.aten.transpose.int,
}


class ScaledGroupedMMTensor(torch.Tensor):
    """
    ScaledGroupedMMTensor is a simple tensor subclass that wraps a regular tensor
    and overrides the torch._grouped_mm op by dispatching to the
    differentiable _scaled_grouped_mm autograd function.
    """

    scaling_type: MoEScalingType = MoEScalingType.FP8_ROWWISE
    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        scaling_type: MoEScalingType,
    ):
        self = torch.Tensor._make_wrapper_subclass(
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
        self.scaling_type = scaling_type
        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        scaling_type: MoEScalingType,
    ):
        self._data = tensor
        self.scaling_type = scaling_type

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # override the grouped mm op to use the differentiable _scaled_grouped_mm
        if func.__name__ == cls.grouped_mm_func_name:
            # Use torchao scaled grouped mm with dynamic quant for
            # "2d x 3d with offsets" case (used for routed experts).
            # Otherwise, fall back to regular grouped mm.
            #
            # TODO: support "3d x 3d without offsets" case, which is
            # used for shared experts. This is basically the grouped_mm
            # kernel handling a bmm.
            A, B = args[0], args[1]
            assert not isinstance(A, ScaledGroupedMMTensor), (
                "A should not be a ScaledGroupedMMTensor"
            )
            assert isinstance(B, ScaledGroupedMMTensor), (
                "B should be a ScaledGroupedMMTensor"
            )
            scaling_type = B.scaling_type
            A_is_2d = A.dim() == 2
            B_is_3d = B.dim() == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None
            other_args = args[2:]
            if A_is_2d and B_is_3d and has_offs:
                return _scaled_grouped_mm(
                    A,
                    B,
                    *other_args,
                    scaling_type=scaling_type,
                    **kwargs,
                )

        # Disable torch_function by hand because we don't want
        # the wrapping behavior of the super() impl, go directly to dispatch
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # detach is special case
        scaling_type = args[0].scaling_type
        if func == torch.ops.aten.detach.default:
            return ScaledGroupedMMTensor(args[0]._data, scaling_type)

        # unwrap args/kwargs
        unwrap = lambda x: x._data if isinstance(x, ScaledGroupedMMTensor) else x
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
            lambda x: ScaledGroupedMMTensor(x, scaling_type),
            out,
        )

    def __repr__(self):
        return f"ScaledGroupedMMTensor(data={self._data}, scaling_type={self.scaling_type})"

    def __tensor_flatten__(self):
        metadata = {"scaling_type": self.scaling_type}
        return ["_data"], metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return ScaledGroupedMMTensor(
            inner_tensors["_data"],
            flatten_spec["scaling_type"],
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
        # cast to mixed precision dtype prior to all-gather
        all_gather_inputs = (self._data.to(mp_policy.param_dtype),)
        all_gather_metadata = ()
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

        # For training step 1+, out=unsharded param.
        if out is not None:
            if isinstance(out, ScaledGroupedMMTensor):
                out_data = out._data
                out.scaling_type = self.scaling_type
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, ScaledGroupedMMTensor
            ):
                out_data = out._local_tensor._data
                out._local_tensor.scaling_type = self.scaling_type
            else:
                raise RuntimeError(
                    f"expect out to be ScaledGroupedMMTensor or DTensor with local_tensor=ScaledGroupedMM, but got {type(out)}"
                )

            # If `data` (all gather outputs) is already in the mixed precision policy param_dtype,
            # verify it has underlying storage as `out` (pre-allocated unsharded param),
            # and then we can just return directly.
            if data.dtype == param_dtype:
                assert (
                    data.untyped_storage().data_ptr()
                    == out_data.untyped_storage().data_ptr()
                )
            else:
                # Otherwise, verify that `out` (pre-allocated unsharded param) has the
                # mixed precision policy param_dtype, then copy `data` to `out`.
                assert out_data.dtype == param_dtype, f"{out_data.dtype} {param_dtype}"
                out_data.copy_(data)

            return

        # For training step 0, out=None, so we need to return a new ScaledGroupedMMTensor.
        output = ScaledGroupedMMTensor(data, self.scaling_type)
        inner_tensors = (data,)
        return output, inner_tensors
