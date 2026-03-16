# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
CuTe DSL copy utilities for vectorized memory operations.

Self-contained copy utility functions extracted from
fbcode/ads_mkl/ops/cute_dsl/quack/copy_utils.py.
"""

from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric],
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
