import torch
import logging

# We use this "hack" to set torchao.__version__ correctly
# For local development this will default to whatever is version.txt
# For release builds this will be set the version+architecture_postfix
import pkg_resources
try:
    __version__ = pkg_resources.get_distribution('torchao').version
except pkg_resources.DistributionNotFound:
    __version__ = 'unknown'


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
