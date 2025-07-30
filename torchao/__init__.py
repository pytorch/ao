import torch
import logging

# torch/nested/_internal/nested_tensor.py:417: UserWarning: Failed to initialize NumPy: No module named 'numpy'
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy: No module named 'numpy'")


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
    except:
        _C = None
        logging.info("Skipping import of cpp extensions")

# Lazy imports to speed up module loading
def __getattr__(name):
    if name == "autoquant":
        from torchao.quantization import autoquant
        return autoquant
    elif name == "quantize_":
        from torchao.quantization import quantize_
        return quantize_
    elif name == "ops":
        from . import ops
        return ops
    elif name == "dtypes":
        from . import dtypes
        return dtypes
    elif name == "testing":
        from . import testing
        return testing
    elif name == "quantization":
        from . import quantization
        return quantization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "dtypes",
    "autoquant",
    "quantize_",
    "testing",
    "quantization",
]

# test-pytorchbot
# test-codev
