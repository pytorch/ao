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

from torchao.prototype.moe_training.config import (
    MXFP8TrainingConfig,
    TrainingBaseConfig,
)
from torchao.prototype.moe_training.utils import _quantize_then_scaled_grouped_mm
from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

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
    torch.ops.aten.t.default,
}


class TorchAOTrainingTensor(TorchAOBaseTensor):
    """
    A subclass of torch.Tensor that overrides the grouped_mm and linear ops
    to use dynamic quantization then low precision grouped_mm/linear op,
    based on the training config.
    """

    config: TrainingBaseConfig = None
    grouped_mm_func_name = "_grouped_mm"
    mm_func_names = ("mm", "matmul", "linear")
    offs_arg_name = "offs"

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        config: TrainingBaseConfig,
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
        self.config = config
        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        config: TrainingBaseConfig,
    ):
        self._data = tensor
        self.config = config

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # grouped_mm op override
        if func.__name__ == cls.grouped_mm_func_name:
            # Use torchao scaled grouped mm with dynamic quant for
            # "2d x 3d with offsets" case (used for routed experts).
            # Otherwise, fall back to regular grouped mm.
            #
            # TODO: support "3d x 3d without offsets" case, which is
            # used for shared experts. This is basically the grouped_mm
            # kernel handling a bmm.
            A, B = args[0], args[1]

            assert not isinstance(A, TorchAOTrainingTensor), (
                "A should not be a TorchAOTrainingTensor"
            )

            assert isinstance(B, TorchAOTrainingTensor), (
                "B should be a TorchAOTrainingTensor"
            )

            config = B.config
            A_is_2d = A.ndim == 2
            B_is_2d_or_3d = B.ndim == 2 or B.ndim == 3
            offs = kwargs.get(cls.offs_arg_name, None)

            if A_is_2d and B_is_2d_or_3d and offs is not None:
                return _quantize_then_scaled_grouped_mm(
                    A,
                    B,
                    offs=offs,
                    config=config,
                )

        # linear op override
        elif func.__name__ in cls.mm_func_names:
            A, B = args[0], args[1]
            assert not isinstance(A, TorchAOTrainingTensor), (
                "A should not be a TorchAOTrainingTensor"
            )

            assert isinstance(B, TorchAOTrainingTensor), (
                "B should be a TorchAOTrainingTensor"
            )

            config = B.config

            if isinstance(config, MXFP8TrainingConfig):
                return _to_mxfp8_then_scaled_mm(
                    A,
                    B,
                    kernel_preference=config.kernel_preference,
                    scale_calculation_mode=config.scale_calculation_mode,
                    wgrad_with_hp=config.wgrad_with_hp,
                )
            else:
                # TODO: support fp8 linear
                raise ValueError(f"unknown config type {type(config)}")

        # Disable torch_function by hand because we don't want
        # the wrapping behavior of the super() impl, go directly to dispatch
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # unwrap args/kwargs and extract config
        config = None

        def unwrap(t):
            nonlocal config
            if config is None:
                config = t.config
            else:
                assert t.config == config, (
                    "All TorchAOTrainingTensor instances must have the same config"
                )
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            TorchAOTrainingTensor, unwrap, (args, kwargs or {})
        )
        assert config is not None, (
            f"__torch_dispatch__ called on {func.__name__} without any TorchAOTrainingTensor arguments"
        )

        # detach is special case
        if func == torch.ops.aten.detach.default:
            return TorchAOTrainingTensor(args_unwrapped[0], config)

        # perform op
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # wrap outputs back into TorchAOTrainingTensor for ops that do preserve subclass
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: TorchAOTrainingTensor(x, config),
            out,
        )

    def __repr__(self):
        return f"TorchAOTrainingTensor(data={self._data}, config={self.config})"

    def __tensor_flatten__(self):
        metadata = {
            "config": self.config,
        }
        return ["_data"], metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return TorchAOTrainingTensor(
            inner_tensors["_data"],
            flatten_spec["config"],
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
            if isinstance(out, TorchAOTrainingTensor):
                out_data = out._data
                out.config = self.config
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, TorchAOTrainingTensor
            ):
                out_data = out._local_tensor._data
                out._local_tensor.config = self.config
            else:
                raise RuntimeError(
                    f"expect out to be TorchAOTrainingTensor or DTensor with local_tensor=ScaledGroupedMM, but got {type(out)}"
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

        # For training step 0, out=None, so we need to return a new TorchAOTrainingTensor.
        output = TorchAOTrainingTensor(data, self.config)
        inner_tensors = (data,)
        return output, inner_tensors
