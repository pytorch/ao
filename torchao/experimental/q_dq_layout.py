# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TODO: delete this file.
# File is kept in torchao/experimental to avoid breaking existing code
import logging

logging.warning(
    "torchao.experimental.q_dq_layout.py is deprecated and will be removed.  Please use torchao.dtypes.uintx.q_dq_layout.py instead."
)
from torchao.dtypes import QDQLayout

__all__ = [
    "QDQLayout",
]
