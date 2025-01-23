import logging
import math
from typing import Optional, Tuple

import torch

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
from torchao.dtypes.utils import (
    Layout,
    PlainLayout,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_and_quantize_affine_hqq,
)

logger = logging.getLogger(__name__)
aten = torch.ops.aten

__all__ = [
    "to_hqq_quantized_intx",
]


class HQQTensor(AffineQuantizedTensor):
    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        zero_point_domain: Optional[ZeroPointDomain] = ZeroPointDomain.INT,
        _layout: Layout = PlainLayout(),
    ):
        input_float = _layout.pre_process(input_float)
        assert (
            zero_point_domain == ZeroPointDomain.FLOAT
            and mapping_type == MappingType.ASYMMETRIC
            and quant_min == 0
        ), "Invalid input parameters for HQQ quantization."
        nbits = int(math.log2(quant_max + 1))
        axis = 1 if (block_size[0] == 1) else 0
        group_size = max(block_size)
        compute_dtype = (
            zero_point_dtype if (zero_point_dtype is not None) else input_float.dtype
        )
        device = input_float.device
        from torchao.dtypes.uintx import TensorCoreTiledLayout

        data, _ = choose_qparams_and_quantize_affine_hqq(
            input_float,
            nbits=nbits,
            group_size=group_size,
            axis=axis,
            compute_dtype=compute_dtype,
            device=device,
            verbose=False,
            raw_output=not isinstance(_layout, (TensorCoreTiledLayout, PlainLayout)),
            # raw_output=False is basically the 'convert to TensorCoreTiledLayout zero_point version' option (add scale*midpoint)
            # note in choose_qparams_affine, preserve_zero = False does this same thing while also controlling whether
            # zero is preserved.
            # TODO uncouple preserve_zero and conversion of zero_point to TensorCoreTiledLayout version
            # TODO move the conversion of zero_point out of quant_primitives and into TensorCoreTiledLayout.from_plain
            # TODO change PlainLayout to use raw_output.
        )
        data = data.to(target_dtype)


to_hqq_quantized_intx = HQQTensor.from_hp_to_intx
