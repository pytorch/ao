import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    Float8Layout,
    Float8AQTTensorImpl,
    register_aqt_quantized_linear_dispatch,
    register_layout,
)
from torchao.float8.inference import (
    Float8MMConfig,
    addmm_float8_unwrapped_inference,
)
from torchao.dtypes.utils import Layout, get_out_shape

from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SemiSparseFloat8Layout(Layout):
    mm_config: Optional[Float8MMConfig] = None

@register_layout(SemiSparseFloat8Layout)
class SemiSparseFloat8AQTTensorImpl(Float8AQTTensorImpl):
    """
    TensorImpl storage class for semi_sparse_cusparselt layout for affine quantized tensor
    """
    def get_plain(self):
        # Currently we don't have cuSPARSELt expansion routines, so we matmul by
        # the identity matrix to get the original dense matrix. This is slow though.
        cols = self.float8_data.numel() * 16 // (10 * self.shape[0])
        float_data_expanded = torch._cslt_sparse_mm(self.float8_data,
                                                    torch.eye(cols,
                                                            dtype=self.float8_data.dtype,
                                                            device=self.float8_data.device).t())
        return float_data_expanded, self.scale, None

    @classmethod
    def from_plain(
        cls,
        float8_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, SemiSparseFloat8Layout)
        float8_data_compressed = torch._cslt_compress(float8_data)
        output = cls(float8_data_compressed, scale, False, _layout)
        return output

def _linear_fp8_act_fp8_weight_semi_structured_check(
    input_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    weight_tensor: Union[torch.Tensor, AffineQuantizedTensor],
    bias: Optional[torch.Tensor],
) -> bool:
    def check_aqt(aqt: Union[torch.Tensor, AffineQuantizedTensor], layout=Float8Layout) -> bool:
        return (
            isinstance(aqt, AffineQuantizedTensor) and
            isinstance(aqt._layout, layout)
            and aqt.tensor_impl.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            and (aqt.shape == aqt.block_size or _is_rowwise_scaled(aqt))
        )
    return check_aqt(input_tensor) and check_aqt(weight_tensor, layout=SemiSparseFloat8Layout)

def _linear_fp8_act_fp8_weight_semi_structured_impl(
    input_tensor: AffineQuantizedTensor,
    weight_tensor: AffineQuantizedTensor,
    bias: Optional[torch.Tensor],
):
    """Implements matmul between FP8 input and FP8 weight with compute using _cslt_sparse_mm"""
    out_shape = get_out_shape(input_tensor.shape, weight_tensor.shape)

    # Weight tensor preprocessing
    scaled_mm_config = weight_tensor._layout.mm_config
    w_tensor_impl = weight_tensor.tensor_impl
    w_data = w_tensor_impl.float8_data
    w_scale = w_tensor_impl.scale

    # Input tensor preprocessing
    inpt_data = input_tensor.tensor_impl.float8_data
    input_scale = input_tensor.tensor_impl.scale
    # Handle case where input tensor is more than 2D
    inpt_data = inpt_data.reshape(-1, inpt_data.shape[-1])

    with torch.no_grad():
        w_compressed = SparseSemiStructuredTensorCUSPARSELT(
            weight_tensor.shape, packed=w_data,
            meta=None, packed_t=None, meta_t=None,
            compressed_swizzled_bitmask=None, requires_grad=False, fuse_transpose_cusparselt=True)

        # Perform the computation
        return addmm_float8_unwrapped_inference(
            inpt_data,
            input_scale,
            w_compressed,
            w_scale,
            output_dtype=input_tensor.dtype,
            bias=bias,
        ).reshape(out_shape)

register_aqt_quantized_linear_dispatch(
    _linear_fp8_act_fp8_weight_semi_structured_check,
    _linear_fp8_act_fp8_weight_semi_structured_impl
)
