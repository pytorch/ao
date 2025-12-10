# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing from torchao.dtypes.uintx.uintx_layout is deprecated. "
    "Please use 'from torchao.prototype.dtypes import UintxLayout, UintxTensor' instead. "
    "This import path will be removed in a future release of torchao. "
    "See https://github.com/pytorch/ao/issues/2752 for more details.",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.dtypes.uintx.uintx_layout import (  # noqa: F401
    _BIT_WIDTH_TO_DTYPE,  # noqa: F401
    _DTYPE_TO_BIT_WIDTH,  # noqa: F401
    UintxAQTTensorImpl,  # noqa: F401
    UintxLayout,  # noqa: F401
    UintxTensor,  # noqa: F401
    to_uintx,  # noqa: F401
)
