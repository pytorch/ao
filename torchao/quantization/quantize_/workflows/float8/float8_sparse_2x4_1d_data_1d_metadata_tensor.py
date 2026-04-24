# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch

from torchao.float8.inference import (
    FP8Granularity,
)
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _quantize_affine_float8,
)
from torchao.quantization.quantize_.common import (
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.utils import get_block_size
from torchao.utils import (
    TorchAOBaseTensor,
)

__all__ = [
    "Float8Sparse2x4_1DData1DMetadataTensor",
]

aten = torch.ops.aten


from .float8_tensor import QuantizeTensorToFloat8Kwargs


class Float8Sparse2x4_1DData1DMetadataTensor(TorchAOBaseTensor):
    """
    Float8 Quantized + 2:4 sparse (weight) Tensor using hipSPARSELt kernels (ROCm/AMD only),
    with float8 dynamic quantization for activation.

    Unlike the CUTLASS variant which stores specified values and metadata as two
    separate tensors, this variant packs them into a single tensor via torch._cslt_compress
    (which dispatches to hipSPARSELt on ROCm), and dispatches matmul via torch._cslt_sparse_mm.

    Only per-tensor scaling is supported — hipSPARSELt lacks per-operand scaling,
    so A_scale * B_scale is passed as the alpha parameter to _cslt_sparse_mm.

    Tensor Attributes:
        packed: compressed sparse tensor (specified values + metadata in one tensor)
        scale: per-tensor scale for float8 quantization

    Non-Tensor Attributes:
        block_size (List[int]): the block size for float8 quantization
        original_shape (Tuple[int, int]): the shape of the original dense weight
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for activation quantization
        dtype: Original Tensor dtype
    """

    tensor_data_names = ["packed", "scale"]
    tensor_attribute_names = []
    optional_tensor_attribute_names = [
        "block_size",
        "original_shape",
        "act_quant_kwargs",
        "dtype",
    ]

    def __new__(
        cls,
        packed: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        original_shape: Optional[Tuple[int, int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        assert original_shape is not None, "original_shape must be specified"
        kwargs = {}
        kwargs["device"] = packed.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        original_shape: Optional[Tuple[int, int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.packed = packed
        self.scale = scale
        self.block_size = block_size
        self.original_shape = original_shape
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.packed.shape=}, {self.scale=}, "
            f"{self.block_size=}, "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    def _quantization_type(self):
        return f"{self.act_quant_kwargs=}, {self.block_size=}, {self.scale.shape=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        in_features = self.shape[1]
        mm_out_dtype = torch.float32
        identity = torch.eye(in_features, device=self.packed.device).to(
            self.packed.dtype
        )
        dense = torch._cslt_sparse_mm(
            self.packed,
            identity,
            alpha=self.scale,
            out_dtype=mm_out_dtype,
        )

        if output_dtype is not None:
            dense = dense.to(output_dtype)
        return dense

    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
        granularity: FP8Granularity = PerTensor(),
        hp_value_lb: Optional[float] = None,
        hp_value_ub: Optional[float] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
    ):
        block_size = get_block_size(hp_tensor.shape, granularity)
        block_size = list(block_size)
        scale = _choose_scale_float8(
            hp_tensor,
            float8_dtype=float8_dtype,
            block_size=block_size,
            hp_value_lb=hp_value_lb,
            hp_value_ub=hp_value_ub,
        )
        data = _quantize_affine_float8(hp_tensor, scale, float8_dtype)
        hp_dtype = hp_tensor.dtype
        original_shape = tuple(hp_tensor.shape)

        assert isinstance(granularity, PerTensor), (
            "hipSPARSELt sparse kernel only supports per-tensor quantization"
        )
        assert float8_dtype == torch.float8_e4m3fn, (
            "hipSPARSELt sparse kernel only supports float8_e4m3fn dtype"
        )

        # Compress using hipSPARSELt — packs specified values + metadata into one tensor
        packed = torch._cslt_compress(data)

        return Float8Sparse2x4_1DData1DMetadataTensor(
            packed,
            scale,
            block_size=block_size,
            original_shape=original_shape,
            act_quant_kwargs=act_quant_kwargs,
            dtype=hp_dtype,
        )


implements = Float8Sparse2x4_1DData1DMetadataTensor.implements
implements_torch_function = (
    Float8Sparse2x4_1DData1DMetadataTensor.implements_torch_function
)


@implements(aten.linear.default)
@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor = kwargs.get("input", args[0] if len(args) > 0 else None)
    weight_tensor = kwargs.get("weight", args[1] if len(args) > 1 else None)
    bias = kwargs.get("bias", args[2] if len(args) > 2 else None)

    assert input_tensor is not None, "input tensor must not be None"
    assert weight_tensor is not None, "weight tensor must not be None"

    act_quant_kwargs = weight_tensor.act_quant_kwargs
    # quantize activation, if `act_quant_kwargs` is specified
    if act_quant_kwargs is not None:
        assert not isinstance(input_tensor, TorchAOBaseTensor), (
            "input tensor was already quantized"
        )
        input_tensor = _choose_quant_func_and_quantize_tensor(
            input_tensor, act_quant_kwargs
        )

    input_fp8 = input_tensor.qdata
    input_scale = input_tensor.scale
    weight_packed = weight_tensor.packed
    weight_scale = weight_tensor.scale

    mm_out_dtype = torch.float32

    # linear: y = x @ W^T
    # _cslt_sparse_mm(compressed_A [m,k], dense_B [k,n]) -> [m,n]
    # A = weight (out_features, in_features), B = input^T (in_features, batch)
    orig_shape = input_fp8.shape
    input_2d = input_fp8.reshape(-1, orig_shape[-1])

    # Per-tensor scaling: pass combined scale as alpha
    alpha = input_scale * weight_scale

    # hipSPARSELt requires bias dtype to match out_dtype
    bias_mm = bias.to(mm_out_dtype) if bias is not None else None

    result = torch._cslt_sparse_mm(
        weight_packed,
        input_2d.t(),
        bias=bias_mm,
        alpha=alpha,
        out_dtype=mm_out_dtype,
    )
    # result: (out_features, batch) -> (batch, out_features)
    result = result.t()

    out_dtype = input_tensor.dtype
    result = result.to(out_dtype)

    result = result.reshape(*orig_shape[:-1], result.shape[-1])

    return result


@implements(aten.to.dtype_layout)
def _(func, types, args, kwargs):
    return (
        args[0]
        .dequantize()
        .to(
            *args[1:],
            dtype=kwargs.get("dtype", args[0].dtype),
            device=kwargs.get("device", args[0].device),
        )
    )


@implements(aten.to.dtype)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
    assert dtype is not None, "dtype must not be None"

    return (
        args[0]
        .dequantize()
        .to(
            dtype=dtype,
        )
    )


# Allow a model with Float8Sparse2x4_1DData1DMetadataTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Float8Sparse2x4_1DData1DMetadataTensor])
