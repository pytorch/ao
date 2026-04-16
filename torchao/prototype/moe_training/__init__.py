# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_to_mxfp8_then_scaled_grouped_mm",
    "_to_fp8_rowwise_then_scaled_grouped_mm",
]


def __getattr__(name: str):
    if name == "_to_fp8_rowwise_then_scaled_grouped_mm":
        from torchao.prototype.moe_training.fp8_grouped_mm import (
            _to_fp8_rowwise_then_scaled_grouped_mm,
        )

        return _to_fp8_rowwise_then_scaled_grouped_mm
    if name == "_to_mxfp8_then_scaled_grouped_mm":
        from torchao.prototype.moe_training.mxfp8_grouped_mm import (
            _to_mxfp8_then_scaled_grouped_mm,
        )

        return _to_mxfp8_then_scaled_grouped_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
