import logging
import os
import re

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


def _parse_version(version_string):
    """
    Parse version string representing pre-release with -1

    Examples: "2.5.0.dev20240708+cu121" -> [2, 5, -1], "2.5.0" -> [2, 5, 0]
    """
    # Check for pre-release indicators
    is_prerelease = bool(re.search(r"(git|dev)", version_string))
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_string)
    if match:
        major, minor, patch = map(int, match.groups())
        if is_prerelease:
            patch = -1
        return [major, minor, patch]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")


# users can set env var TORCH_INCOMPATIBLE=1 to skip loading .so files
# this way, if they are using an incompatbile torch version, they can still use the API by setting the env var
skip_loading_so_files = bool(os.getenv("TORCH_INCOMPATIBLE", False))
# if torchao version has "+git", assume it's locally built and we don't know
#   anything about the PyTorch version used to build it
# otherwise, assume it's prebuilt by torchao's build scripts and we can make
#   assumptions about the PyTorch version used to build it.
if skip_loading_so_files:
    logger.warning(
        f"Skipping import of cpp extensions due to incompatible torch version {torch.__version__} for torchao version {__version__} \
        Please see GitHub issue #2919 for more info"
    )
elif not ("+git" in __version__) and not ("unknown" in __version__):
    # We know that torchao .so files built using PyTorch 2.8.0 are not ABI compatible with PyTorch 2.9+. (see #2919)
    # The following code skips importing the .so files if incompatible torch version is detected,
    # to avoid crashing the Python process with "Aborted (core
    # dumped)".
    # .dev for nightlies since 2.9.0 has not been released
    if _parse_version(torch.__version__) == _parse_version(
        "2.9.0.dev"
    ) and _parse_version(__version__) == _parse_version("0.14.0"):
        pass
    else:
        logger.warning(
            f"Skipping import of cpp extensions due to incompatible torch version {torch.__version__} for torchao version {__version__} \
            Please see GitHub issue #2919 for more info"
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

        # The following registers meta kernels for some CPU kernels
        from torchao.csrc_meta_ops import *  # noqa: F403
    except Exception as e:
        logger.debug(f"Skipping import of cpp extensions: {e}")

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
