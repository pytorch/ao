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
        assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
        torch.ops.load_library(str(so_files[0]))
        from . import ops

    # The following library contains CPU kernels from torchao/experimental
    # They are built automatically by ao/setup.py if on an ARM machine.
    # They can also be built outside of the torchao install process by
    # running the script `torchao/experimental/build_torchao_ops.sh <aten|executorch>`
    # For more information, see https://github.com/pytorch/ao/blob/main/torchao/experimental/docs/readme.md
    experimental_lib = list(Path(__file__).parent.glob("libtorchao_ops_aten.*"))
    if len(experimental_lib) > 0:
        assert (
            len(experimental_lib) == 1
        ), f"Expected at most one libtorchao_ops_aten.* file, found {len(experimental_lib)}"
        torch.ops.load_library(str(experimental_lib[0]))
except:
    logging.debug("Skipping import of cpp extensions")

from torchao.quantization import (
    autoquant,
    quantize_,
)

from . import dtypes, optim, testing

__all__ = [
    "dtypes",
    "autoquant",
    "optim",
    "quantize_",
    "testing",
    "ops",
]
