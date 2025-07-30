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
    logger.debug(f"Skipping import of cpp extensions: {e}")

# Lazy imports to speed up module loading
def __getattr__(name):
    if name == "autoquant":
        from torchao.quantization import autoquant
        return autoquant
    elif name == "quantize_":
        from torchao.quantization import quantize_
        return quantize_
    elif name == "dtypes":
        from . import dtypes
        return dtypes
    elif name == "optim":
        from . import optim
        return optim
    elif name == "quantization":
        from . import quantization
        return quantization
    elif name == "swizzle":
        from . import swizzle
        return swizzle
    elif name == "testing":
        from . import testing
        return testing
    elif name == "ops":
        # ops may or may not be available depending on if .so files exist
        if hasattr(__import__(__name__), 'ops'):
            from . import ops
            return ops
        raise AttributeError(f"module {__name__!r} has no attribute 'ops'")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
