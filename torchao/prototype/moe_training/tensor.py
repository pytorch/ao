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
    FP8GroupedMMConfig,
    MXFP8TrainingConfig,
    TrainingBaseConfig,
)
from torchao.prototype.moe_training.fp8_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
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


class MXFP8TrainingTensor(TorchAOBaseTensor):
    """
    MXFP8TrainingTensor is a simple tensor subclass that wraps a regular tensor
    and overrides mm and grouped_mm ops, dispatching to autograd functions that
    dynamically quantize the op inputs to MXFP8:
    differentiable _quantize_then_scaled_grouped_mm autograd function.
    """

    config: MXFP8TrainingConfig = None
    grouped_mm_func_name = "_grouped_mm"
    mm_func_names = ("mm", "matmul", "linear")
    offs_arg_name = "offs"

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        config: MXFP8TrainingConfig,
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
        config: MXFP8TrainingConfig,
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
            assert not isinstance(A, MXFP8TrainingTensor), (
                "A should not be a MXFP8TrainingTensor"
            )
            assert isinstance(B, MXFP8TrainingTensor), (
                "B should be a MXFP8TrainingTensor"
            )
            config = B.config
            A_is_2d = A.ndim == 2
            B_is_2d_or_3d = B.ndim == 2 or B.ndim == 3
            offs = kwargs.get(cls.offs_arg_name, None)
            if A_is_2d and B_is_2d_or_3d and offs is not None:
                return _to_mxfp8_then_scaled_grouped_mm(
                    A,
                    B,
                    offs,
                    out_dtype=config.out_dtype,
                    kernel_preference=config.kernel_preference,
                    wgrad_with_hp=config.wgrad_with_hp,
                    scale_calculation_mode=config.scale_calculation_mode,
                )

        # linear op override
        elif func.__name__ in cls.mm_func_names:
            A, B = args[0], args[1]
            assert not isinstance(A, MXFP8TrainingTensor), (
                "A should not be a MXFP8TrainingTensor"
            )
            assert isinstance(B, MXFP8TrainingTensor), (
                "B should be a MXFP8TrainingTensor"
            )
            config = B.config
            return _to_mxfp8_then_scaled_mm(
                A,
                B,
                kernel_preference=config.kernel_preference,
                scale_calculation_mode=config.scale_calculation_mode,
                wgrad_with_hp=config.wgrad_with_hp,
            )

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
                    "All MXFP8TrainingTensor instances must have the same config"
                )
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            MXFP8TrainingTensor, unwrap, (args, kwargs or {})
        )
        assert config is not None, (
            f"__torch_dispatch__ called on {func.__name__} without any MXFP8TrainingTensor arguments"
        )

        # detach is special case
        if func == torch.ops.aten.detach.default:
            return MXFP8TrainingTensor(args_unwrapped[0], config)

        # perform op
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # wrap outputs back into MXFP8TrainingTensor for ops that do preserve subclass
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: MXFP8TrainingTensor(x, config),
            out,
        )

    def __repr__(self):
        return f"MXFP8TrainingTensor(data={self._data}, config={self.config})"

    def __tensor_flatten__(self):
        metadata = {
            "config": self.config,
        }
        return ["_data"], metadata

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return MXFP8TrainingTensor(
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
            if isinstance(out, MXFP8TrainingTensor):
                out_data = out._data
                out.config = self.config
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, MXFP8TrainingTensor
            ):
                out_data = out._local_tensor._data
                out._local_tensor.config = self.config
            else:
                raise RuntimeError(
                    f"expect out to be MXFP8TrainingTensor or DTensor with local_tensor=ScaledGroupedMM, but got {type(out)}"
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

        # For training step 0, out=None, so we need to return a new MXFP8TrainingTensor.
        output = MXFP8TrainingTensor(data, self.config)
        inner_tensors = (data,)
        return output, inner_tensors


# dispatching helper for MXFP8TrainingTensor
def _quantize_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    config: TrainingBaseConfig,
    offs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function performs dynamic quantization with the given config
    on the input tensors A and B, then performs a scaled grouped GEMM and returns the results.

    Args:
        A (bf16/float32 torch.Tensor): The first high-precision input tensor, which must be a 2D tensor of shape (M * num_groups, K)
            and in row-major memory layout.
        B_t (bf16/float32 torch.Tensor): The second high-precision input tensor which must be 3D, which must be shape (E, K, N)
            and in column-major memory layout.
        offs (int32 torch.Tensor): The offsets to use to mark the starting index of each group along dim0 of the A tensor.
        config (MXFP8TrainingConfig): Configuration for grouped matmul quantization.
    """
    # Dispatch based on derived dtype
    if isinstance(config, FP8GroupedMMConfig):
        return _to_fp8_rowwise_then_scaled_grouped_mm(
            A,
            B_t,
            offs,
            config.out_dtype,
            config.float8_dtype,
        )
    elif isinstance(config, MXFP8TrainingConfig):
        return _to_mxfp8_then_scaled_grouped_mm(
            A,
            B_t,
            offs,
            out_dtype=config.out_dtype,
            kernel_preference=config.kernel_preference,
            wgrad_with_hp=config.wgrad_with_hp,
            scale_calculation_mode=config.scale_calculation_mode,
        )
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
