import logging

# torch/nested/_internal/nested_tensor.py:417: UserWarning: Failed to initialize NumPy: No module named 'numpy'
import warnings

import importlib
import sys
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

logger = logging.getLogger(__name__)

skip_loading_so_files = False
# if torchao version has "+git", assume it's locally built and we don't know
#   anything about the PyTorch version used to build it
# otherwise, assume it's prebuilt by torchao's build scripts and we can make
#   assumptions about the PyTorch version used to build it.
if (not "+git" in __version__) and not ("unknown" in __version__):
    # torchao v0.13.0 is built with PyTorch 2.8.0. We know that torchao .so
    # files built using PyTorch 2.8.0 are not ABI compatible with PyTorch 2.9+.
    # The following code skips importing the .so files if PyTorch 2.9+ is
    # detected, to avoid crashing the Python process with "Aborted (core
    # dumped)".
    # TODO(#2901, and before next torchao release): make this generic for
    # future torchao and torch versions
    if __version__.startswith("0.13.0") and str(torch.__version__) >= "2.9":
        logger.warning(
            f"Skipping import of cpp extensions due to incompatible torch version {torch.__version__} for torchao version {__version__}"
        )
        skip_loading_so_files = True

if not skip_loading_so_files:
    try:
        from pathlib import Path

        so_files = list(Path(__file__).parent.glob("_C*.so"))
        if len(so_files) > 0:
            for file in so_files:
                torch.ops.load_library(str(file))
            from . import ops

        # The following library contains CPU kernels from torchao/experimental
        # They are built automatically by ao/setup.py if on an ARM machine.
        # They can also be built outside of the torchao install process by
        # running the script `torchao/experimental/build_torchao_ops.sh <aten|executorch>`
        # For more information, see https://github.com/pytorch/ao/blob/main/torchao/experimental/docs/readme.md
        # Avoid eagerly importing experimental op_lib as it is heavy and not always needed.
        # Users can trigger it by importing `torchao.experimental` or setting up kernels explicitly.
    except Exception as e:
        logger.debug(f"Skipping import of cpp extensions: {e}")

# Lazy submodule and attribute exposure to reduce import-time overhead
_LAZY_SUBMODULES = {
    "dtypes": "torchao.dtypes",
    "optim": "torchao.optim",
    "quantization": "torchao.quantization",
    "swizzle": "torchao.swizzle",
    "testing": "torchao.testing",
    "ops": "torchao.ops",
    "kernel": "torchao.kernel",
    "float8": "torchao.float8",
    "sparsity": "torchao.sparsity",
    "prototype": "torchao.prototype",
    "experimental": "torchao.experimental",
    "_models": "torchao._models",
    "core": "torchao.core",
}

_LAZY_ATTRS = {
    # Top-level convenience re-exports
    "autoquant": ("torchao.quantization", "autoquant"),
    "quantize_": ("torchao.quantization", "quantize_"),
}

__all__ = [
    # Submodules
    "dtypes",
    "optim",
    "quantization",
    "swizzle",
    "testing",
    "ops",
    "kernel",
    "float8",
    "sparsity",
    "prototype",
    "experimental",
    "_models",
    "core",
    # Attributes
    "autoquant",
    "quantize_",
]

def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        setattr(sys.modules[__name__], name, module)
        return module
    if name in _LAZY_ATTRS:
        mod_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_name)
        value = getattr(module, attr_name)
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
