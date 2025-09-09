# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

import torch

from torchao.quantization.granularity import (
    PerGroup,
)
from torchao.quantization.observer import get_block_size
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _quantize_affine_float8,
)
from torchao.utils import (
    TorchAOBaseTensor,
)

from .float8_tensor import QuantizeTensorToFloat8Kwargs

__all__ = [
    "Float8OpaqueTensor",
]

aten = torch.ops.aten


class Float8OpaqueTensor(TorchAOBaseTensor):
    """
    Float8 dynamic activation float8 weight on CPU. The weight tensor is reordered to a blocked layout
    for better memory locality from [N, K] to [N/block_n, K/block_k, block_k, block_n], where block_n = 32
    and block_k depends on group-size for quantization (=32/64/128). And the innermost block with shape
    [block_k, block_n] may be further reordered to VNNI layout depending on supported CPU ISA.

    Tensor Attributes:
        qdata: Reordered float8 weight on CPU with shape = [N/block_n, K/block_k, block_k, block_n].
        scale: Scale tensor for weight, dtype = float32. For per-group/row quantization, shape =
               [N / block_n, num_groups, block_n]. For per-tensor quantization, shape = [1].

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity. for groupwise quantization,
                    block_size is (1, group_size). we only support group_size = 32/64/128. For per-row
                    quantization, blocks_size is (1, K). For per-tensor quantization, block_size is (N, K).
        shape: shape of the original Tensor
        act_quant_kwargs: the kwargs for from_hp
    """

    tensor_data_names = ["qdata", "scale"]
    tensor_attribute_names = ["block_size", "act_quant_kwargs"]

    def __new__(
        cls,
        qdata,
        scale,
        block_size,
        act_quant_kwargs,
    ):
        if qdata.ndim == 2:
            shape = qdata.shape
        else:
            assert qdata.ndim == 4
            shape = torch.Size(
                [qdata.size(0) * qdata.size(3), qdata.size(1) * qdata.size(2)]
            )
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        block_size: List[int],
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        self.qdata = qdata
        self.scale = scale
        self.block_size = block_size
        self.act_quant_kwargs = act_quant_kwargs

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}, {self.act_quant_kwargs=}"

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: List[int],
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        assert hp_tensor.ndim == 2 and hp_tensor.device.type == "cpu", (
            f"Expecting 2D tensor on CPU, but got: {hp_tensor.shape} on {hp_tensor.device.type}"
        )
        assert len(block_size) == hp_tensor.ndim
        N = hp_tensor.size(0)
        K = hp_tensor.size(-1)
        assert (block_size[0] == 1 or block_size[0] == N) and block_size[1] in (
            32,
            64,
            128,
            K,
        ), f"Unsupported block_size: {block_size} for tensor shape {hp_tensor}"
        # assert N % 32 == 0, (
        #     f"Expecting out_features {N} to be multiple of 32, but got {N}"
        # )
        assert act_quant_kwargs is not None, (
            "Activation quantization args must be provided for Float8OpaqueTensor"
        )
        act_per_group_quant = isinstance(act_quant_kwargs.granularity, PerGroup)
        wei_per_group_quant = block_size[1] < K
        if act_per_group_quant:
            group_size = act_quant_kwargs.granularity.group_size
            if wei_per_group_quant:
                # weight_tensor is also per group quantized
                assert block_size[1] == group_size, (
                    "input and weight should have the same group size but got"
                    f" {block_size[1]} and {group_size}"
                )
        if act_per_group_quant or wei_per_group_quant:
            assert N % 32 == 0, (
                f"Expecting out_features {N} to be multiple of 32, but got {N}"
            )
            assert K % block_size[1] == 0, (
                f"Expecting in_features {K} to be multiple of group_size {block_size[1]}, but got {K}"
            )
        scale = _choose_scale_float8(
            hp_tensor,
            float8_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )
        data = _quantize_affine_float8(hp_tensor, scale, torch.float8_e4m3fn)
        # Pack weight from [N, K] to [N / block_n, K / block_k, block_k, block_n].
        # Pack scales from [N, num_groups] to [N / block_n, num_groups, block_n].
        packed_weight, packed_scale = torch.ops.torchao.float8_linear_prepack_cpu(
            data, scale
        )

        return Float8OpaqueTensor(
            qdata=packed_weight,
            scale=packed_scale,
            block_size=block_size,
            act_quant_kwargs=act_quant_kwargs,
        )


implements = Float8OpaqueTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert input_tensor.device.type == "cpu", (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert isinstance(weight_tensor, Float8OpaqueTensor), (
        f"Expected weight_tensor to be Float8OpaqueTensor, got: {type(weight_tensor)}"
    )
    assert weight_tensor.ndim in [2, 4]
    assert input_tensor.size(-1) == weight_tensor.size(-1), (
        f"Shapes of input and weight do not match, input:{input_tensor.shape}, weight: {weight_tensor.shape}"
    )

    act_mat = input_tensor.contiguous()
    packed_weight = weight_tensor.qdata
    scale = weight_tensor.scale

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype
    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # activation float8 quantization
    if (
        weight_tensor.act_quant_kwargs is not None
        and weight_tensor.act_quant_kwargs.granularity is not None
    ):
        granularity = weight_tensor.act_quant_kwargs.granularity
        if isinstance(granularity, PerGroup):
            group_size = granularity.group_size
            if weight_tensor.block_size[1] < weight_tensor.size(-1):
                # weight_tensor is also per group quantized
                assert weight_tensor.block_size[1] == group_size, (
                    "input and weight should have the same group size but got"
                    f" {weight_tensor.block_size[1]} and {group_size}"
                )
        act_block_size = get_block_size(act_mat.shape, granularity)
        act_scale = _choose_scale_float8(
            act_mat,
            float8_dtype=torch.float8_e4m3fn,
            block_size=act_block_size,
        )
        act_mat = _quantize_affine_float8(act_mat, act_scale, torch.float8_e4m3fn)
    else:
        raise NotImplementedError(
            "Activation quantization args not provided for Float8OpaqueTensor"
        )

    # float8 quantized linear operation
    y = torch.ops.torchao.float8_linear_cpu.default(
        act_mat,
        act_scale,
        packed_weight,
        scale,
        bias.float() if bias is not None else bias,  # requires bias to be float
        torch.float,  # out_dtype
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    return y.to(orig_dtype)


Float8OpaqueTensor.__module__ = "torchao.quantization"

# Allow a model with Float8OpaqueTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Float8OpaqueTensor])
