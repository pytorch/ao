# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch


def _is_column_major(x: torch.Tensor) -> bool:
    """
    This function checks if the input tensor is column-major.

    Args:
        x (torch.Tensor): The input tensor to be checked.

    Returns:
        A boolean indicating whether the input tensor is column-major.
    """
    assert x.ndim == 2 or x.ndim == 3, "input tensor must be 2D or 3D"
    return x.stride(-2) == 1 and x.stride(-1) > 1
