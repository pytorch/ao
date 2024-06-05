import torch
import logging

_IS_FBCODE = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and
    torch._utils_internal.IS_FBSOURCE
)

if not _IS_FBCODE:
    try:
        from . import _C
        from . import ops
    except:
        _C = None
        logging.info("Skipping import of cpp extensions")

from torchao.quantization import (
    apply_weight_only_int8_quant,
    apply_dynamic_quant,
    autoquant,
)
from . import dtypes

from torchao.kv_cache import PagedAttentionCache, PagedTensor
__all__ = [
    "dtypes",
    "apply_dynamic_quant",
    "apply_weight_only_int8_quant",
    "autoquant",
    "PagedAttentionCache",
    "PagedTensor"
]
