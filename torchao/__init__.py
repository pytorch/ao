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
    if __version__.startswith("0.13.0") and torch.__version__ > "2.8":
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
        from torchao.experimental.op_lib import *  # noqa: F403
    except Exception as e:
        logger.warning(f"Skipping import of cpp extensions: {e}")

from torchao.quantization import (
    autoquant,
    quantize_,
)

from . import dtypes, optim, quantization, swizzle, testing

__all__ = [
    "dtypes",
    "autoquant",
    "optim",
    "quantize_",
    "swizzle",
    "testing",
    "ops",
    "quantization",
]
