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

try:
    from pathlib import Path

    so_files = list(Path(__file__).parent.glob("_C*.so"))
    if len(so_files) > 0:
        compute_capability = (
            torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        )

        for file in so_files:
            # only load architecture-specific target if the current GPU matches that target
            if (
                ("cutlass_90a" in file.name and compute_capability != (9, 0))
                or ("cutlass_100a" in file.name and compute_capability != (10, 0))
                or ("cutlass_120a" in file.name and compute_capability != (12, 0))
            ):
                continue

            torch.ops.load_library(str(file))

        from . import ops

    # The following library contains CPU kernels from torchao/experimental
    # They are built automatically by ao/setup.py if on an ARM machine.
    # They can also be built outside of the torchao install process by
    # running the script `torchao/experimental/build_torchao_ops.sh <aten|executorch>`
    # For more information, see https://github.com/pytorch/ao/blob/main/torchao/experimental/docs/readme.md
    from torchao.experimental.op_lib import *  # noqa: F403
except Exception as e:
    logging.debug(f"Skipping import of cpp extensions: {e}")

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
