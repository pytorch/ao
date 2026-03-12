# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import List, Optional

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_qparams_affine_tinygemm,
    _choose_qparams_and_quantize_affine_hqq,
    _quantize_affine_tinygemm,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.quantization.quantize_.workflows import (
    Int4ChooseQParamsAlgorithm,
)
from torchao.quantization.utils import (
    _get_per_token_block_size,
    pack_tinygemm_scales_and_zeros,
)
from torchao.utils import (
    TorchAOBaseTensor,
    torch_version_at_least,
)

__all__ = [
    "Int4OpaqueTensor",
]

aten = torch.ops.aten


class Int4OpaqueTensor(TorchAOBaseTensor):
    """
    int4 weight quantization on CPU with two supported paths:

    1. A16W4 (weight-only): float16/bfloat16/float32 activation + int4 weight,
       using tinygemm kernel (_weight_int4pack_mm_for_cpu).

    2. DA8W4 (dynamic activation): int8 dynamic activation + int4 weight,
       using da8w4 kernel (da8w4_linear_cpu). Activation is quantized
       per-token dynamically at runtime.

    The path is selected based on `act_mapping_type`:
      - None  → A16W4 tinygemm path
      - "asymmetric" or "symmetric" → DA8W4 path

    Mandatory Tensor Attributes (A16W4):
        qdata: packed int4 weight.
               A16W4: preshuffled for tinygemm, shape (N, K/2)
               DA8W4: prepacked for da8w4_linear_cpu (4D)
        scale_and_zero: weight quantization params.
               A16W4: packed scales+zeros for tinygemm, shape (K/group_size, N, 2)
               DA8W4: packed scales, shape (N/block_n, G, block_n)

    Mandatory Non-Tensor Attributes:
        block_size: quantization block size, e.g. [1, group_size]
        shape: original weight shape [N, K]

    Optional Tensor Data Attributes:
        act_pre_scale: activation pre-scale (A16W4 only)
        qzeros: packed weight zero-points for DA8W4, shape (N/block_n, G, block_n)
        compensation: weight compensation for DA8W4, shape (N/block_n, K/block_k, block_n)

    Optional Non-Tensor Attributes:
        act_mapping_type: None for A16W4; "asymmetric" or "symmetric" for DA8W4
    """

    tensor_data_names = ["qdata", "scale_and_zero"]
    tensor_attribute_names = ["block_size", "shape"]
    optional_tensor_data_names = ["act_pre_scale", "qzeros", "compensation"]
    optional_tensor_attribute_names = ["act_mapping_type"]

    def __new__(
        cls,
        qdata,
        scale_and_zero,
        block_size,
        shape,
        act_pre_scale: Optional[torch.Tensor] = None,
        qzeros: Optional[torch.Tensor] = None,
        compensation: Optional[torch.Tensor] = None,
        act_mapping_type: Optional[str] = None,
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale_and_zero.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata: torch.Tensor,
        scale_and_zero: torch.Tensor,
        block_size: List[int],
        shape: torch.Size,
        act_pre_scale: Optional[torch.Tensor] = None,
        qzeros: Optional[torch.Tensor] = None,
        compensation: Optional[torch.Tensor] = None,
        act_mapping_type: Optional[str] = None,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale_and_zero = scale_and_zero
        self.block_size = block_size
        self.act_pre_scale = act_pre_scale
        self.qzeros = qzeros
        self.compensation = compensation
        self.act_mapping_type = act_mapping_type

    def _quantization_type(self):
        if self.act_mapping_type is not None:
            s = f"da8w4, shape={self.shape}, block_size={self.block_size}, act={self.act_mapping_type}, device={self.device}"
        else:
            s = f"shape={self.shape}, block_size={self.block_size}, device={self.device}"
            if self.act_pre_scale is not None:
                s += f", act_pre_scale.shape={self.act_pre_scale.shape}"
        return s

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
        int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = Int4ChooseQParamsAlgorithm.TINYGEMM,
    ):
        assert w.ndim == 2 and w.device.type == "cpu", (
            f"Expecting 2D tensor on CPU, but got: {w.shape} on {w.device.type}"
        )
        assert len(block_size) == w.ndim
        assert block_size[0] == 1 and block_size[1] in (32, 64, 128), (
            f"Expecting groupwise quantization with group size = 32/64/128, but got block_size: {block_size}"
        )
        original_shape = w.shape
        mapping_type = MappingType.ASYMMETRIC
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        scale_dtype = None
        zero_point_dtype = w.dtype

        # we support two paths for constructing a Int4OpaqueTensor
        # 1. use [hqq](https://mobiusml.github.io/hqq_blog/) algorithm to compute
        # scale and zero_point, then convert to the format that's compatible with tinygemm kernels
        # 2. don't use hqq, use default tinygemm algorithm to compute scale and zero_point
        #
        # both approach should have the same performance since both are using CPU tinygemm kernel for gemm
        # 1. typically will have higher accuracy compared to 2.
        if int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.HQQ:
            nbits = int(math.log2(quant_max + 1))
            axis = 1
            group_size = block_size[-1]
            int_data, scale, zero_point, _ = _choose_qparams_and_quantize_affine_hqq(
                w,
                nbits=nbits,
                group_size=group_size,
                axis=axis,
                compute_dtype=zero_point_dtype,
                device=w.device,
            )
            int_data = int_data.to(target_dtype)
        else:
            assert (
                int4_choose_qparams_algorithm == Int4ChooseQParamsAlgorithm.TINYGEMM
            ), (
                f"Unsupported Int4ChooseQParamsAlgorithm: {int4_choose_qparams_algorithm}"
            )

            scale, zero_point = _choose_qparams_affine_tinygemm(
                w,
                mapping_type,
                block_size,
                target_dtype,
                quant_min,
                quant_max,
                eps,
                scale_dtype,
                zero_point_dtype,
            )
            int_data = _quantize_affine_tinygemm(
                w,
                block_size,
                scale,
                zero_point,
                target_dtype,
                quant_min,
                quant_max,
            )
        assert int_data.dtype == torch.int32, (
            "torch.ops.aten._convert_weight_to_int4pack_for_cpu expects `int32` dtype"
        )
        packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            int_data,
            1,  # innerKTiles is not needed for CPU
        )

        scale = scale.reshape(int_data.shape[0], -1)
        zero_point = zero_point.reshape(int_data.shape[0], -1)

        scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero_point, scale.dtype)
        return Int4OpaqueTensor(
            qdata=packed_weight,
            scale_and_zero=scale_and_zero,
            block_size=block_size,
            shape=original_shape,
            act_pre_scale=None,
        )

    @classmethod
    def from_hp_da8w4(
        cls,
        w: torch.Tensor,
        group_size: int = 32,
        act_mapping_type: MappingType = MappingType.ASYMMETRIC,
    ):
        """
        Quantize a float weight tensor for DA8W4 (int8 dynamic activation + int4 weight) on CPU.

        The weight is quantized per-group (asymmetric int4), then prepacked via
        torch.ops.torchao.da8w4_linear_prepack_cpu for the CPU kernel.

        Args:
            w: float weight tensor, shape [N, K], must be on CPU
            group_size: quantization group size, K must be divisible by group_size
            act_mapping_type: MappingType.ASYMMETRIC (uint8 activation, default) or
                              MappingType.SYMMETRIC (int8 activation, requires PyTorch >= 2.8)
        """
        assert w.ndim == 2 and w.device.type == "cpu", (
            f"Expecting 2D tensor on CPU, but got: {w.shape} on {w.device.type}"
        )
        assert w.shape[1] % group_size == 0, (
            f"K={w.shape[1]} must be divisible by group_size={group_size}"
        )
        assert w.shape[0] % 32 == 0 and w.shape[1] % 2 == 0, (
            f"N={w.shape[0]} must be divisible by 32 and K={w.shape[1]} must be even for DA8W4"
        )
        original_shape = w.shape
        block_size = [1, group_size]

        # Quantize weight: asymmetric int4 per-group → uint8 [N, K], values in [0, 15]
        scale, zero_point = choose_qparams_affine(
            w,
            MappingType.ASYMMETRIC,
            block_size,
            torch.uint8,
            quant_min=0,
            quant_max=15,
            eps=1e-6,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int32,
        )
        int4_weight = quantize_affine(
            w,
            block_size,
            scale,
            zero_point,
            torch.uint8,
            quant_min=0,
            quant_max=15,
        ).to(torch.uint8)

        # Prepack for da8w4_linear_cpu
        packed_weight, packed_scales, packed_qzeros, compensation = (
            torch.ops.torchao.da8w4_linear_prepack_cpu(
                int4_weight,
                scale,
                zero_point.to(torch.int8),
            )
        )

        act_str = (
            "symmetric" if act_mapping_type == MappingType.SYMMETRIC else "asymmetric"
        )
        return cls(
            qdata=packed_weight,
            scale_and_zero=packed_scales,
            block_size=block_size,
            shape=original_shape,
            act_pre_scale=None,
            qzeros=packed_qzeros,
            compensation=compensation,
            act_mapping_type=act_str,
        )


implements = Int4OpaqueTensor.implements
implements_torch_function = Int4OpaqueTensor.implements_torch_function


def _da8w4_linear(input_tensor, weight_tensor, bias):
    """DA8W4 linear: dynamically quantize activation per-token, then call da8w4_linear_cpu."""
    orig_act_size = input_tensor.size()
    orig_dtype = input_tensor.dtype

    # Reshape activation to 2D
    act_fp = input_tensor.reshape(-1, input_tensor.shape[-1])
    per_token_block_size = _get_per_token_block_size(act_fp)

    if weight_tensor.act_mapping_type == "symmetric":
        assert torch_version_at_least("2.8.0"), (
            "Symmetric int8 activation quantization requires PyTorch 2.8+"
        )
        # Symmetric int8 quantization: values in [-127, 127]
        act_scale, act_zero_point = choose_qparams_affine(
            act_fp,
            MappingType.SYMMETRIC,
            per_token_block_size,
            torch.int8,
            quant_min=-127,
            quant_max=127,
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int8,
        )
        act_int = quantize_affine(
            act_fp,
            per_token_block_size,
            act_scale,
            act_zero_point,
            torch.int8,
            quant_min=-127,
            quant_max=127,
        )
    else:
        assert torch_version_at_least("2.7.0"), (
            "Asymmetric uint8 activation quantization requires PyTorch 2.7+"
        )
        # Asymmetric uint8 quantization: values in [0, 255]
        act_scale, act_zero_point = choose_qparams_affine(
            act_fp,
            MappingType.ASYMMETRIC,
            per_token_block_size,
            torch.uint8,
            quant_min=0,
            quant_max=255,
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int32,
        )
        act_int = quantize_affine(
            act_fp,
            per_token_block_size,
            act_scale,
            act_zero_point,
            torch.uint8,
            quant_min=0,
            quant_max=255,
        )

    act_scale_1d = act_scale.reshape(-1)
    act_qzeros_1d = act_zero_point.reshape(-1).to(torch.int32)

    y = torch.ops.torchao.da8w4_linear_cpu.default(
        act_int.contiguous(),
        act_scale_1d,
        act_qzeros_1d,
        weight_tensor.qdata,
        weight_tensor.scale_and_zero,
        weight_tensor.qzeros,
        weight_tensor.compensation,
        bias.float() if bias is not None else bias,
        orig_dtype,
    )

    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)
    return y.to(orig_dtype)


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert input_tensor.device.type == "cpu", (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert isinstance(weight_tensor, Int4OpaqueTensor), (
        f"Expected weight_tensor to be Int4OpaqueTensor, got: {type(weight_tensor)}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"Shapes of input and weight do not match, input:{input_tensor.shape}, weight: {weight_tensor.shape}"
    )

    # DA8W4 path: dynamic int8 activation + int4 weight
    if weight_tensor.act_mapping_type is not None:
        if weight_tensor.act_mapping_type == "symmetric":
            assert torch_version_at_least("2.8.0"), (
                "Symmetric int8 activation quantization requires PyTorch 2.8+"
            )
        else:
            assert torch_version_at_least("2.7.0"), (
                "Asymmetric uint8 activation quantization requires PyTorch 2.7+"
            )
        return _da8w4_linear(input_tensor, weight_tensor, bias)

    # A16W4 path: float activation + int4 weight (tinygemm)
    if weight_tensor.act_pre_scale is not None:
        input_tensor = input_tensor * weight_tensor.act_pre_scale

    act_mat = input_tensor
    packed_weight = weight_tensor.qdata
    scale_and_zero = weight_tensor.scale_and_zero

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act_mat = act_mat.reshape(-1, act_mat.shape[-1])

    # groupwise int4 quantization
    groupsize = weight_tensor.block_size[1]
    y = torch.ops.aten._weight_int4pack_mm_for_cpu(
        act_mat.contiguous(), packed_weight, groupsize, scale_and_zero
    )

    # remove out_feature padding
    assert weight_tensor.ndim == 2
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    if bias is not None:
        y += bias
    return y.to(orig_dtype)


Int4OpaqueTensor.__module__ = "torchao.prototype.int4_opaque_tensor"

# Allow a model with Int4OpaqueTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4OpaqueTensor])
