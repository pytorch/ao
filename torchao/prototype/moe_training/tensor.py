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

from torchao.prototype.moe_training import _quantize_then_scaled_grouped_mm
from torchao.prototype.moe_training.conversion_utils import (
    GroupedMMConfig,
    MoEScalingType,
)
from torchao.quantization.quantize_.common import KernelPreference

logger: logging.Logger = logging.getLogger(__name__)

_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,  # for *.to(dtype)
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
}


class ScaledGroupedMMTensor(torch.Tensor):
    """
    ScaledGroupedMMTensor is a simple tensor subclass that wraps a regular tensor
    and overrides the torch._grouped_mm op by dispatching to the
    differentiable _quantize_then_scaled_grouped_mm autograd function.
    """

    scaling_type: MoEScalingType = MoEScalingType.FP8_ROWWISE
    kernel_preference: KernelPreference = KernelPreference.AUTO
    grouped_mm_func_name = "_grouped_mm"
    offs_arg_name = "offs"

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        scaling_type: MoEScalingType,
        grouped_mm_config: GroupedMMConfig,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
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
        self.kernel_preference = kernel_preference
        self.grouped_mm_config = grouped_mm_config
        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        scaling_type: MoEScalingType,
        grouped_mm_config: GroupedMMConfig,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
    ):
        self._data = tensor
        self.scaling_type = scaling_type
        self.kernel_preference = kernel_preference
        self.grouped_mm_config = grouped_mm_config

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # override the grouped mm op to use the differentiable _quantize_then_scaled_grouped_mm
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
            kernel_preference = B.kernel_preference
            grouped_mm_config = B.grouped_mm_config

            A_is_2d = A.ndim == 2
            B_is_2d_or_3d = B.ndim == 2 or B.ndim == 3
            has_offs = kwargs.get(cls.offs_arg_name) is not None
            other_args = args[2:]

            if A_is_2d and B_is_2d_or_3d and has_offs:
                return _quantize_then_scaled_grouped_mm(
                    A,
                    B,
                    *other_args,
                    scaling_type=scaling_type,
                    kernel_preference=kernel_preference,
                    grouped_mm_config=grouped_mm_config,
                    **kwargs,
                )

        # Disable torch_function by hand because we don't want
        # the wrapping behavior of the super() impl, go directly to dispatch
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # unwrap args/kwargs and extract scaling_type and kernel_preference
        scaling_type = None
        kernel_preference = None
        grouped_mm_config = None

        def unwrap(t):
            nonlocal scaling_type, kernel_preference, grouped_mm_config
            if scaling_type is None:
                scaling_type = t.scaling_type
                kernel_preference = t.kernel_preference
                grouped_mm_config = t.grouped_mm_config
            else:
                assert t.scaling_type == scaling_type
                assert t.kernel_preference == kernel_preference
                assert t.grouped_mm_config == grouped_mm_config
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            ScaledGroupedMMTensor, unwrap, (args, kwargs or {})
        )
        assert scaling_type is not None, (
            f"__torch_dispatch__ called on {func.__name__} without any ScaledGroupedMMTensor arguments"
        )

        # detach is special case
        if func == torch.ops.aten.detach.default:
            return ScaledGroupedMMTensor(
                args_unwrapped[0], scaling_type, grouped_mm_config, kernel_preference
            )

        # perform op
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # wrap outputs back into ScaledGroupedMMTensor for ops that do preserve subclass
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: ScaledGroupedMMTensor(
                x, scaling_type, grouped_mm_config, kernel_preference
            ),
            out,
        )

    def __repr__(self):
        return f"ScaledGroupedMMTensor(data={self._data}, scaling_type={self.scaling_type}, grouped_mm_config={self.grouped_mm_config}, kernel_preference={self.kernel_preference})"

    def __tensor_flatten__(self):
        metadata = {
            "scaling_type": self.scaling_type,
            "kernel_preference": self.kernel_preference,
            "grouped_mm_config": self.grouped_mm_config,
        }
        return ["_data"], metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return ScaledGroupedMMTensor(
            inner_tensors["_data"],
            flatten_spec["scaling_type"],
            flatten_spec["grouped_mm_config"],
            flatten_spec["kernel_preference"],
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
                out.grouped_mm_config = self.grouped_mm_config
                out.kernel_preference = self.kernel_preference
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, ScaledGroupedMMTensor
            ):
                out_data = out._local_tensor._data
                out._local_tensor.scaling_type = self.scaling_type
                out._local_tensor.grouped_mm_config = self.grouped_mm_config
                out._local_tensor.kernel_preference = self.kernel_preference
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
        output = ScaledGroupedMMTensor(
            data, self.scaling_type, self.grouped_mm_config, self.kernel_preference
        )
        inner_tensors = (data,)
        return output, inner_tensors
