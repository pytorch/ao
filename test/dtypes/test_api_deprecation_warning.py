# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for deprecated API imports that have been moved to prototype.
TODO: Remove these tests once the deprecated APIs have been removed.
"""

import sys
import warnings


def test_int8_dynamic_act_int4_weight_cpu_layout_deprecated():
    """Test deprecation warning for Int8DynamicActInt4WeightCPULayout."""
    # We need to clear the cache to force re-importing and trigger the warning again.
    modules_to_clear = [
        "torchao.dtypes.uintx.dyn_int8_act_int4_wei_cpu_layout",
        "torchao.dtypes",
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    with warnings.catch_warnings(record=True) as w:
        from torchao.dtypes import Int8DynamicActInt4WeightCPULayout  # noqa: F401

        warnings.simplefilter("always")  # Ensure all warnings are captured
        assert any(
            issubclass(warning.category, DeprecationWarning)
            and "Int8DynamicActInt4WeightCPULayout" in str(warning.message)
            for warning in w
        ), (
            f"Expected deprecation warning for Int8DynamicActInt4WeightCPULayout, got: {[str(warning.message) for warning in w]}"
        )


def test_cutlass_int4_packed_layout_deprecated():
    """Test deprecation warning for CutlassInt4PackedLayout."""
    # We need to clear the cache to force re-importing and trigger the warning again.
    modules_to_clear = [
        "torchao.dtypes.uintx.cutlass_int4_packed_layout",
        "torchao.dtypes",
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    with warnings.catch_warnings(record=True) as w:
        from torchao.dtypes import CutlassInt4PackedLayout  # noqa: F401

        warnings.simplefilter("always")  # Ensure all warnings are captured
        assert any(
            issubclass(warning.category, DeprecationWarning)
            and "CutlassInt4PackedLayout" in str(warning.message)
            for warning in w
        ), (
            f"Expected deprecation warning for CutlassInt4PackedLayout, got: {[str(warning.message) for warning in w]}"
        )


def test_block_sparse_layout_deprecated():
    """Test deprecation warning for BlockSparseLayout."""
    # We need to clear the cache to force re-importing and trigger the warning again.
    modules_to_clear = [
        "torchao.dtypes.uintx.block_sparse_layout",
        "torchao.dtypes",
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    with warnings.catch_warnings(record=True) as w:
        from torchao.dtypes import BlockSparseLayout  # noqa: F401

        warnings.simplefilter("always")  # Ensure all warnings are captured
        assert any(
            issubclass(warning.category, DeprecationWarning)
            and "BlockSparseLayout" in str(warning.message)
            for warning in w
        ), (
            f"Expected deprecation warning for BlockSparseLayout, got: {[str(warning.message) for warning in w]}"
        )


def test_marlin_qqq_layout_deprecated():
    """Test deprecation warning for MarlinQQQLayout."""
    # We need to clear the cache to force re-importing and trigger the warning again.
    modules_to_clear = [
        "torchao.dtypes.uintx.marlin_qqq_tensor",
        "torchao.dtypes",
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Ensure all warnings are captured
        from torchao.dtypes import MarlinQQQLayout  # noqa: F401

        assert any(
            issubclass(warning.category, DeprecationWarning)
            and "MarlinQQQLayout" in str(warning.message)
            for warning in w
        ), (
            f"Expected deprecation warning for MarlinQQQLayout, got: {[str(warning.message) for warning in w]}"
        )
