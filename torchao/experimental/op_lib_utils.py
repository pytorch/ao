# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def _check_torchao_ops_loaded():
    # Check kernels are installed/loaded
    try:
        torch.ops.torchao._pack_8bit_act_4bit_weight
    except AttributeError:
        raise Exception(
            "TorchAO experimental kernels are not loaded.  To install the kernels, run `USE_CPP=1 pip install .` from ao on a machine with an ARM CPU."
            + " You can also set target to 'aten' if you are using ARM CPU."
        )
