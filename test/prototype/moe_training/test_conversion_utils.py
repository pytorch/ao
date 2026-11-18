# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
from torchao.prototype.moe_training.conversion_utils import _get_tensor_cls_for_config
from torchao.prototype.moe_training.tensor import MXFP8TrainingWeightWrapperTensor


def test_get_tensor_cls_for_config_mxfp8_works_without_nightly_symbols():
    """MXFP8 training conversion must not hard-require torch-nightly private symbols.

    `torch.distributed.tensor._ops.scaled_mm_single_dim_strategy` /
    `_dispatch.is_pinned_handler` are only needed for the DTensor-based scaled_mm
    dispatch paths; the conversion itself must succeed on stable torch builds.
    """
    assert (
        _get_tensor_cls_for_config(MXFP8TrainingOpConfig())
        is MXFP8TrainingWeightWrapperTensor
    )
