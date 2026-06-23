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


def is_fbcode():
    return not hasattr(torch.version, "git_version")


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


skip_loading_so_files = False
_skip_reason = None
force_skip_loading_so_files = (
    os.getenv("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "0") == "1"
)
if force_skip_loading_so_files:
    # user override
    # users can set env var TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 to skip loading .so files
    # this way, if they are using an incompatbile torch version, they can still use the API by setting the env var
    skip_loading_so_files = True
    _skip_reason = (
        "Skipping import of cpp extensions due to TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1"
    )
elif is_fbcode():
    skip_loading_so_files = False
# if torchao version has "+git", assume it's locally built and we don't know
#   anything about the PyTorch version used to build it unless user provides override flag
# otherwise, assume it's prebuilt by torchao's build scripts and we can make
#   assumptions about the PyTorch version used to build it.
elif not ("+git" in __version__) and not ("unknown" in __version__):
    current_torch_version = _parse_version(torch.__version__)
    min_torch_version = _parse_version("2.11.0")
    if current_torch_version >= min_torch_version:
        skip_loading_so_files = False
    else:
        skip_loading_so_files = True
        _skip_reason = (
            f"Skipping import of cpp extensions due to incompatible torch version. "
            f"Please upgrade to torch >= 2.11.0 (found {torch.__version__}). "
            f"Set TORCHAO_SILENCE_VERSION_COMPATIBILITY_WARNING=1 to silence this warning."
        )

if skip_loading_so_files:
    if not os.getenv("TORCHAO_SILENCE_VERSION_COMPATIBILITY_WARNING", "0") == "1":
        logger.warning(_skip_reason)
else:
    try:
        from pathlib import Path

        # Load .so files
        so_files = list(Path(__file__).parent.glob("_C*.so"))
        if len(so_files) > 0:
            for file in so_files:
                logger.debug(f"Loading {file}")
                try:
                    torch.ops.load_library(str(file))
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
            from . import ops

        # The following registers meta kernels for some CPU kernels
        from torchao.csrc_meta_ops import *  # noqa: F403
    except Exception as e:
        logger.debug(f"Skipping import of cpp extensions: {e}")

# must import dtypes before quantization
from . import dtypes  # noqa: I001

from torchao.quantization import (
    quantize_,
)

from . import optim, quantization, swizzle, testing

__all__ = [
    "dtypes",
    "optim",
    "quantize_",
    "swizzle",
    "testing",
    "ops",
    "quantization",
]
