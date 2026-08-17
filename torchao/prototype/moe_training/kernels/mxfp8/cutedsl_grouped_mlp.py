# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher facade for the MXFP8 routed-expert grouped-MLP CuTe DSL kernels.

``grouped_mlp_ops`` imports its three launchers from this module and from
nowhere else, so this name and these three signatures are the seam between the
custom-op layer and the kernels. Keep it a pure re-export: anything defined here
would be code the ops layer depends on but that no kernel test covers.

The import is deliberately per-launcher rather than a package-level ``*``, so a
kernel that has not landed yet costs an ImportError naming that kernel instead of
breaking the two that have.
"""

from torchao.prototype.moe_training.kernels.mxfp8.kernel_wgrad import (
    launch_grouped_gemm_wgrad,
)

__all__ = ["launch_grouped_gemm_wgrad"]
