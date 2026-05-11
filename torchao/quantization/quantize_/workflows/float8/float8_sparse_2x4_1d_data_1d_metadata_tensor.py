# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch
from torch.library import custom_op
from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

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
    is_MI350,
    is_ROCM,
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
        qdata_and_metadata: compressed sparse tensor (specified values + metadata in one tensor)
        scale: per-tensor scale for float8 quantization

    Non-Tensor Attributes:
        block_size (List[int]): the block size for float8 quantization
        original_shape (Tuple[int, int]): the shape of the original dense weight
        act_quant_kwargs (QuantizeTensorToFloat8Kwargs): the kwargs for activation quantization
        dtype: Original Tensor dtype
    """

    tensor_data_names = ["qdata_and_metadata", "scale"]
    tensor_attribute_names = []
    optional_tensor_attribute_names = [
        "block_size",
        "original_shape",
        "act_quant_kwargs",
        "dtype",
    ]

    def __new__(
        cls,
        qdata_and_metadata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        original_shape: Optional[Tuple[int, int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        assert original_shape is not None, "original_shape must be specified"
        kwargs = {}
        kwargs["device"] = qdata_and_metadata.device
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        qdata_and_metadata: torch.Tensor,
        scale: torch.Tensor,
        block_size: Optional[List[int]] = None,
        original_shape: Optional[Tuple[int, int]] = None,
        act_quant_kwargs: Optional[QuantizeTensorToFloat8Kwargs] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.qdata_and_metadata = qdata_and_metadata
        self.scale = scale
        self.block_size = block_size
        self.original_shape = original_shape
        self.act_quant_kwargs = act_quant_kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.act_quant_kwargs=}, {self.qdata_and_metadata.shape=}, {self.scale=}, "
            f"{self.block_size=}, "
            f"{self.shape=}, {self.device=}, {self.dtype=})"
        )

    def _quantization_type(self):
        return f"{self.act_quant_kwargs=}, {self.block_size=}, {self.scale.shape=}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        in_features = self.shape[1]
        mm_out_dtype = torch.float32
        identity = torch.eye(in_features, device=self.qdata_and_metadata.device).to(
            self.qdata_and_metadata.dtype
        )
        dense = torch._cslt_sparse_mm(
            self.qdata_and_metadata,
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
        assert is_ROCM(), (
            "SPARSE_1D_DATA_1D_METADATA packing format is only supported on ROCm (hipSPARSELt)"
        )
        assert is_MI350(), "hipSPARSELt FP8 sparse requires MI350 (gfx950) GPU"
        assert isinstance(granularity, PerTensor), (
            "hipSPARSELt sparse kernel only supports per-tensor quantization"
        )
        assert float8_dtype == torch.float8_e4m3fn, (
            "hipSPARSELt sparse kernel only supports float8_e4m3fn dtype"
        )

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

        # Compress using hipSPARSELt — packs specified values + metadata into one tensor
        qdata_and_metadata = torch._cslt_compress(data)

        return Float8Sparse2x4_1DData1DMetadataTensor(
            qdata_and_metadata,
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


@custom_op("float8_sparse::fp8_sparse_mm", mutates_args=())
def fp8_sparse_mm(
    dense: torch.Tensor,
    packed: torch.Tensor,
    bias: Optional[torch.Tensor],
    alpha: torch.Tensor,
    out_features: int,
    min_size: int,
) -> torch.Tensor:
    """Padded sparse matmul for hipSPARSELt with per-tensor FP8 scaling.

    Wraps torch._cslt_sparse_mm with batch-dim padding so that torch.export
    does not see a data-dependent shape change.
    """
    batch, k = dense.shape
    to_pad = (-batch) % min_size
    if to_pad:
        dense = torch.nn.functional.pad(dense, (0, 0, 0, to_pad))
    result = torch._cslt_sparse_mm(
        packed,
        dense.t(),
        bias=bias,
        alpha=alpha,
        out_dtype=torch.float32,
    )
    result = result.t()
    if to_pad:
        result = result.narrow(0, 0, batch)
    return result.contiguous()


@fp8_sparse_mm.register_fake
def _fp8_sparse_mm_fake(
    dense: torch.Tensor,
    packed: torch.Tensor,
    bias: Optional[torch.Tensor],
    alpha: torch.Tensor,
    out_features: int,
    min_size: int,
) -> torch.Tensor:
    batch = dense.shape[0]
    return torch.empty(
        batch,
        out_features,
        dtype=torch.float32,
        device=dense.device,
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
    weight_packed = weight_tensor.qdata_and_metadata
    weight_scale = weight_tensor.scale

    # linear: y = x @ W^T
    # _cslt_sparse_mm(compressed_A [m,k], dense_B [k,n]) -> [m,n]
    # A = weight (out_features, in_features), B = input^T (in_features, batch)
    orig_shape = input_fp8.shape
    input_2d = input_fp8.reshape(-1, orig_shape[-1])

    constraints = SparseSemiStructuredTensorCUSPARSELT._DTYPE_SHAPE_CONSTRAINTS[
        input_2d.dtype
    ]

    # Per-tensor scaling: pass combined scale as alpha
    alpha = input_scale * weight_scale

    # hipSPARSELt requires bias dtype to match out_dtype (float32)
    bias_mm = bias.to(torch.float32) if bias is not None else None

    result = torch.ops.float8_sparse.fp8_sparse_mm(
        input_2d,
        weight_packed,
        bias_mm,
        alpha,
        weight_tensor.original_shape[0],
        # dense_min_rows == dense_min_cols == 16 for float8_e4m3fn
        constraints.dense_min_rows,
    )

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
