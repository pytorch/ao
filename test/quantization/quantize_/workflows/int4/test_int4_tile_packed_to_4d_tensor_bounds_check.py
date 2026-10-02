# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4572.

`Int4TilePackedTo4dTensor` stored `block_size` with no validation against the
actual packed `qdata`/`scale_and_zero` tensors. At inference,
`groupsize = block_size[-1]` is handed straight to
`torch.ops.aten._weight_int4pack_mm(act, qdata, groupsize, scale_and_zero)`,
which trusts `groupsize` to be consistent with how many (scale, zero) groups
`scale_and_zero` actually holds. A checkpoint whose `block_size` metadata was
edited to declare a smaller (but still individually valid) groupsize than the
one `qdata`/`scale_and_zero` were actually packed with makes the tinygemm
kernel iterate more groups than `scale_and_zero` holds, reading out of bounds
(this is unreachable via `from_hp`, which always derives `block_size` from
the tensor it packs -- the vulnerable path is deserializing a tampered/
inconsistent checkpoint that skips `from_hp`, e.g. via `__tensor_unflatten__`
or direct construction).

The real tinygemm ops (`_convert_weight_to_int4pack` / `_weight_int4pack_mm`)
only have CUDA kernels registered, but the packed *shapes* they produce are
fully determined by simple arithmetic and are reproducible with `device="meta"`
tensors (verified below against the real op via the `Meta` dispatch key) --
so this test constructs `Int4TilePackedTo4dTensor` instances directly with
correctly- and incorrectly-shaped meta tensors, with no GPU involved.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
    Int4TilePackedTo4dTensor,
)

# Fixed for this packing format, see the class docstring.
_INNER_K_TILES = 8


def _make_qdata_and_scale_and_zero(n, k, group_size, num_experts=None):
    """Build correctly-shaped, empty meta qdata/scale_and_zero tensors for an
    (n, k) weight (or a batch of `num_experts` of them), matching the shapes
    `Int4TilePackedTo4dTensor.from_hp` actually produces (confirmed against
    the real `_convert_weight_to_int4pack` op run on `device="meta"`, since
    it only has CUDA kernels registered otherwise).
    """
    packed_n, packed_k_groups = n // 8, k // (_INNER_K_TILES * 16)
    num_groups = k // group_size
    if num_experts is None:
        qdata_shape = (packed_n, packed_k_groups, 32, _INNER_K_TILES // 2)
        sz_shape = (num_groups, n, 2)
    else:
        qdata_shape = (num_experts, packed_n, packed_k_groups, 32, _INNER_K_TILES // 2)
        sz_shape = (num_experts, num_groups, n, 2)
    qdata = torch.zeros(qdata_shape, dtype=torch.int32, device="meta")
    scale_and_zero = torch.zeros(sz_shape, dtype=torch.bfloat16, device="meta")
    return qdata, scale_and_zero


class TestInt4TilePackedTo4dTensorBoundsCheck(TestCase):
    def test_packed_qdata_shape_matches_real_op(self):
        """Sanity-check `_make_qdata_and_scale_and_zero`'s shape formula
        against the real `_convert_weight_to_int4pack` op (Meta dispatch key
        is the only backend-independent one registered for it)."""
        n, k = 64, 1024
        packed = torch.ops.aten._convert_weight_to_int4pack(
            torch.zeros((n, k // 2), dtype=torch.uint8, device="meta"),
            _INNER_K_TILES,
        )
        qdata, _ = _make_qdata_and_scale_and_zero(n, k, group_size=128)
        self.assertEqual(packed.shape, qdata.shape)

    def test_consistent_block_size_constructs_fine(self):
        for n, k, group_size in [(64, 1024, 128), (256, 2048, 64), (32, 256, 32)]:
            qdata, scale_and_zero = _make_qdata_and_scale_and_zero(n, k, group_size)
            # should not raise
            Int4TilePackedTo4dTensor(
                qdata, scale_and_zero, [1, group_size], torch.Size((n, k))
            )

    def test_moe_consistent_block_size_constructs_fine(self):
        num_experts, n, k, group_size = 4, 64, 1024, 128
        qdata, scale_and_zero = _make_qdata_and_scale_and_zero(
            n, k, group_size, num_experts=num_experts
        )
        # should not raise
        Int4TilePackedTo4dTensor(
            qdata,
            scale_and_zero,
            [1, 1, group_size],
            torch.Size((num_experts, n, k)),
        )

    def test_tampered_smaller_groupsize_is_rejected(self):
        """The issue's exact scenario: qdata/scale_and_zero packed with
        group_size=128, but block_size metadata edited to declare a smaller,
        individually-valid groupsize (32) without re-packing. Previously this
        was accepted silently and would have made `_weight_int4pack_mm` read
        past the end of `scale_and_zero`."""
        n, k = 64, 1024
        qdata, scale_and_zero = _make_qdata_and_scale_and_zero(
            n, k, group_size=128
        )
        with self.assertRaisesRegex(RuntimeError, "is inconsistent with"):
            Int4TilePackedTo4dTensor(
                qdata, scale_and_zero, [1, 32], torch.Size((n, k))
            )

    def test_tampered_larger_groupsize_is_rejected(self):
        n, k = 64, 1024
        qdata, scale_and_zero = _make_qdata_and_scale_and_zero(
            n, k, group_size=32
        )
        with self.assertRaisesRegex(RuntimeError, "is inconsistent with"):
            Int4TilePackedTo4dTensor(
                qdata, scale_and_zero, [1, 128], torch.Size((n, k))
            )

    def test_mismatched_scale_and_zero_num_groups_is_rejected(self):
        """Directly truncated scale_and_zero (fewer groups than qdata/
        block_size imply), independent of any particular groupsize edit."""
        n, k, group_size = 64, 1024, 128
        qdata, scale_and_zero = _make_qdata_and_scale_and_zero(n, k, group_size)
        truncated_scale_and_zero = scale_and_zero[:-1]  # drop the last group
        with self.assertRaisesRegex(RuntimeError, "is inconsistent with"):
            Int4TilePackedTo4dTensor(
                qdata, truncated_scale_and_zero, [1, group_size], torch.Size((n, k))
            )


if __name__ == "__main__":
    run_tests()
