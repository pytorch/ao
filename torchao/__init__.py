import torch
from torch.testing._internal.common_utils import IS_FBCODE
if not IS_FBCODE:
    from . import _C
    from . import ops

from torchao.quantization import (
    apply_weight_only_int8_quant,
    apply_dynamic_quant,
    autoquant,
)
from . import dtypes

__all__ = [
    "dtypes",
    "apply_dynamic_quant",
    "apply_weight_only_int8_quant",
    "autoquant",
]
