# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear` which
does not require `torch.compile` to be performant..
"""

import torch


class Float8LinearNoCompile(torch.nn.Linear):
    """
    Float8LinearNoCompile is a version of Float8Linear that does not require
    the use of torch.compile to be performant.

    Note: this is **prototype** and not suitable for production use.
    """

    pass
