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
    autoquant,
)
from . import dtypes

__all__ = [
    "dtypes",
    "autoquant",
]
