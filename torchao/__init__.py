import logging

# torch/nested/_internal/nested_tensor.py:417: UserWarning: Failed to initialize NumPy: No module named 'numpy'
import warnings

import torch

warnings.filterwarnings(
    "ignore", message="Failed to initialize NumPy: No module named 'numpy'"
)


# We use this "hack" to set torchao.__version__ correctly
# the version of ao is dependent on environment variables for multiple architectures
# For local development this will default to whatever is version.txt
# For release builds this will be set the version+architecture_postfix
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchao")
except PackageNotFoundError:
    __version__ = "unknown"  # In case this logic breaks don't break the build

_IS_FBCODE = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and torch._utils_internal.IS_FBSOURCE
)
if not _IS_FBCODE:
    try:
        from pathlib import Path

        so_files = list(Path(__file__).parent.glob("_C*.so"))
        assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
        torch.ops.load_library(so_files[0])
        from . import ops
    except:
        logging.debug("Skipping import of cpp extensions")

from torchao.quantization import (
    autoquant,
    quantize_,
)

from . import dtypes, testing

__all__ = [
    "dtypes",
    "autoquant",
    "quantize_",
    "testing",
    "ops",
]

# test-pytorchbot
# test-codev
