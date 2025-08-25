# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Protocols for some functionalities in tensor subclasses"""

from typing import Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class SupportsActivationScaling(Protocol):
    """Protocol for activation scale, using for algorithms like AWQ
    A class that have `act_scale: Optional[torch.Tensor]` attribute implements the Protocol
    """

    act_scale: Optional[torch.Tensor]
