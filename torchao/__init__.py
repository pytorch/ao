from torchao.quantization import (
    apply_weight_only_int8_quant,
    apply_dynamic_quant,
    autoquant,
)
from . import dtypes
import torch
_IS_FBCODE = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and
    torch._utils_internal.IS_FBSOURCE
)

if not _IS_FBCODE:
    from . import _C
    from . import ops

__all__ = [
    "dtypes",
    "apply_dynamic_quant",
    "apply_weight_only_int8_quant",
    "autoquant",
]
