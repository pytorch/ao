# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    quantize_affine,
)
from torchao.utils import TorchAOBaseTensor

__all__ = [
    "Int4MarlinSparseTensor",
]

aten = torch.ops.aten


try:
    from fbgemm_gpu.experimental.gen_ai.quantize import int4_row_quantize_zp, pack_int4
except:
    int4_row_quantize_zp = None
    pack_int4 = None


class Int4MarlinSparseTensor(TorchAOBaseTensor):
    tensor_data_names = ["qdata", "scale", "zero_point", "meta"]  # meta is a tensor
    tensor_attribute_names = ["block_size", "num_bits", "shape"]

    def __new__(cls, qdata, scale, zero_point, meta, block_size, num_bits, shape):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self, qdata, scale, zero_point, meta, block_size, num_bits, shape
    ):  # args need to match lines 37 and 38
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.meta = meta
        self.block_size = block_size
        self.num_bits = num_bits

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}"

    @staticmethod
    def pre_process(input: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor to be in the correct format for the Marlin sparse kernel.
            - 1º: the input tensor is transposed since the linear layer keeps the weights in a transposed format
            - 2º: tensor is injected with 2:4 sparsity
            - 3º: transposes it again because the quantization process will compute the scales for dim=-1

        Args:
            input (torch.Tensor): the input tensor to preprocess

        Returns:
            torch.Tensor: the preprocessed tensor
        """
        from torchao.sparsity.marlin import inject_24  # avoid circular import

        input_t = input.t()
        w_24, _ = inject_24(input_t, *input_t.shape)
        return w_24.t()

    @classmethod
    def from_plain(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ):
        from torchao.sparsity.marlin import const, pack_to_marlin_24

        # Linear layers are (in_features, out_features) but the qdata that is reaching this point
        # is (out_features, in_features). We need to transpose it to match the expected shape in the marlin code.
        q_w_24 = qdata.t()
        # addressing the case when scale has dimension 1, happens when
        # weight_shape[-1] == group_size == 128
        if scale.ndim == 1:
            scale = scale.reshape(scale.shape[0], -1)

        scale_t = scale.t()

        if not torch.cuda.get_device_capability()[0] >= 8:
            raise ValueError(
                f"Can not use Sparse Marlin 2:4 int4*fp16 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel."
            )

        if q_w_24.dtype != torch.int32:
            raise ValueError("Only `torch.int32` weights are supported.")

        in_features, out_features = q_w_24.shape
        if in_features % 128 != 0 or out_features != 256 == 0:
            raise ValueError(
                "`in_features` must be divisible by 64 and `out_features` by 256."
            )

        # NOTE: The current marlin 2:4 kernel supports both 4 and 8 bits quantization but fp8
        # will require a bit more work to get our current quantization flow to work with it.
        # Check the link for a reference: https://github.com/neuralmagic/nm-vllm/tree/main
        num_bits = 4 if torch.max(q_w_24) < 16 else -1
        if num_bits not in [4]:
            raise ValueError(f"Only {[4]} bits are supported, got {num_bits}.")

        group_size = in_features // scale_t.shape[0]
        if group_size == 0:
            group_size = in_features
        assert group_size <= in_features, (
            "Group size must be less than or equal to in_features."
        )

        if group_size not in const.SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Only {const.SUPPORTED_GROUP_SIZES} group sizes are supported, got {group_size}."
            )

        # Compress quantized weight to marlin 2:4 format
        marlin_24_q_w_comp, marlin_24_s, meta = pack_to_marlin_24(
            q_w_24, scale_t, num_bits, group_size
        )

        return cls(
            qdata=marlin_24_q_w_comp,
            scale=marlin_24_s,
            zero_point=zero_point,
            meta=meta,
            block_size=group_size,
            shape=q_w_24.shape,
            num_bits=num_bits,
        )

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: List[int],
    ):
        preprocessed_w = cls.pre_process(w)
        assert block_size[-1] == 128 or block_size[-1] == preprocessed_w.shape[-1], (
            f"MarlinSparse only supports 128 group size or per channel quantization, got {block_size}"
        )

        quant_min = 0
        quant_max = 15
        target_dtype = torch.int32

        scale, zero_point = choose_qparams_affine(
            input=preprocessed_w,
            mapping_type=MappingType.SYMMETRIC,
            block_size=block_size,
            target_dtype=target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=1e-6,
        )

        wq = quantize_affine(
            input=preprocessed_w,
            block_size=block_size,
            scale=scale,
            zero_point=zero_point,
            output_dtype=target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
        )

        scale = scale.to(w.dtype)
        zero_point = zero_point.to(w.dtype)

        return cls.from_plain(qdata=wq, scale=scale, zero_point=zero_point)


implements = Int4MarlinSparseTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    from torchao.ops import marlin_24_gemm
    from torchao.sparsity.marlin import marlin_24_workspace

    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale.is_contiguous(), "Expected scale to be contiguous"
    assert weight_tensor.zero_point.is_contiguous(), (
        "Expected zero_point to be contiguous"
    )

    sparse_w_int4 = weight_tensor.qdata
    scale = weight_tensor.scale
    meta = weight_tensor.meta
    original_shape = weight_tensor.shape
    num_bits = weight_tensor.num_bits

    # Folds batch dimension into the first dimension
    input_2d = input_tensor.view(-1, input_tensor.shape[-1])

    size_m = input_2d.shape[0]
    size_n = scale.shape[1]
    size_k = input_2d.shape[1]
    workspace_24 = marlin_24_workspace(original_shape[1])

    out = marlin_24_gemm(
        input_2d,
        sparse_w_int4,
        meta,
        scale,
        workspace_24,
        num_bits,
        size_m,
        size_n,
        size_k,
    )

    # Unfold the batch dimension
    out = out.reshape(input_tensor.shape[:-1] + (scale.shape[1],))

    if bias is not None:
        out += bias.to(out.dtype)
    return out
