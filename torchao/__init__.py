import torch
import logging

# We use this "hack" to set torchao.__version__ correctly
# the version of ao is dependent on environment variables for multiple architectures
# For local development this will default to whatever is version.txt
# For release builds this will be set the version+architecture_postfix
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("torchao")
except PackageNotFoundError:
    __version__ = 'unknown'  # In case this logic breaks don't break the build

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
    quantize,
    register_apply_tensor_subclass,
)
from . import dtypes

__all__ = [
    "dtypes",
    "autoquant",
    "quantize",
    "register_apply_tensor_subclass",
]
