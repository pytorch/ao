# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    MappingType,
    quantize_affine,
)

from torchao.utils import fill_defaults, TORCH_VERSION_AT_LEAST_2_5, TorchAOBaseTensor


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
    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape", "meta", "num_bits"]

    def __new__(cls, qdata, scale, shape):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(self, qdata, scale, zero_point, meta, block_size, shape, num_bits):
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.meta = meta
        self.block_size = block_size
        self.shape = shape
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
        zero: torch.Tensor,
    ):
        from torchao.sparsity.marlin import const, pack_to_marlin_24

        # Linear layers are (in_features, out_features) but the int_data that is reaching this point
        # is (out_features, in_features). We need to transpose it to match the expected shape in the marlin code.
        q_w_24 = int_data.t()
        # addressing the case when scale has dimension 1, happens when
        # weight_shape[-1] == group_size == 128
        if scale.ndim == 1:
            scale = scale.reshape(scale.shape[0], -1)

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: Tuple[int],  # quantize functions needs it as tuple not list
    ):
        preprocessed_w = cls.pre_process(w)
        # assert (
        #     len(block_size) == w.ndim
        # ), f"Expecting the length of block_size to be equal to the dimension of the weight, got {block_size=} and {w.ndim=}"
        # if int4_row_quantize_zp is None:
        #     raise ImportError("Requires fbgemm-gpu-genai >= 1.2.0")

        # assert (
        #     all(x == 1 for x in block_size[:-1]) and block_size[-1] != 1
        # ), "Only groupwise quant is supported right now"

        # group_size = block_size[-1]
        # original_shape = w.shape

        assert (
            block_size == 128 or block_size == w.shape[-1]
        ), f"MarlinSparseLayout only supports 128 group size or per channel quantization, got {block_size}"

        quant_min = 0
        quant_max = 15
        target_dtype = torch.int4

        scale, zero_point = choose_qparams_affine(
            input=preprocessed_w,
            mapping_type=MappingType.SYMMETRIC,
            block_size=block_size,
            target_dtype=torch.int4,  # ??? i think its int4 because we wanna convert to int4 idk but in the old version i think its int32
            quant_min=quant_min,
            quant_max=quant_max,
            eps=1e-6,
            # leaving scale dtype and zero point dtype as default for now idk
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


implements = Int4MarlinSparseTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale.is_contiguous(), "Expected scale to be contiguous"
    assert (
        weight_tensor.zero_point.is_contiguous()
    ), "Expected zero_point to be contiguous"

    orig_act_size = input_tensor.size()
    orig_out_features = weight_tensor.shape[-2]

    input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
    res = torch.ops.fbgemm.bf16i4bf16_rowwise(
        input_tensor,
        weight_tensor.qdata,
        weight_tensor.scale,
        weight_tensor.zero_point,
    )
    res = res.reshape(*orig_act_size[:-1], orig_out_features)
    if bias is not None:
        res = res + bias
    return res
