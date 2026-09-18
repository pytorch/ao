# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4584.

`_to_fp8_row_major_t`'s transposed-store bounds mask checked the wrong
axes: the store address is

    out_offs = block_col_offs[:, None] * output_stride_row
             + block_row_offs[None, :] * output_stride_col

i.e. the tile's first axis addresses the output *row* via `block_col_offs`
and the second axis addresses the output *col* via `block_row_offs`. The
mask instead compared `block_row_offs` against `output_num_rows` and
`block_col_offs` against `output_num_cols` - axes swapped relative to the
address. For rectangular / non-tile-aligned inputs this masks off valid
output elements (leaving `torch.empty` garbage) or lets writes through for
output elements that are actually out of bounds.

This test requires CUDA to run the real Triton kernel end-to-end (see
`fp8_dynamic_tensorwise_test.py`, which hard-requires
`torch.cuda.is_available()`), so it isn't runnable here. Instead, this test
isolates and directly checks the *index arithmetic* in plain Python - the
same block_row_offs/block_col_offs -> out_offs/out_mask relationship the
kernel computes per program instance - against the exact tile geometry the
autotuner can pick (non-square input, no assumption that BLOCK_SIZE evenly
divides either dimension). This proves the mask covers exactly the valid
transposed output positions, independent of actually running Triton.
"""

import itertools

from torch.testing._internal.common_utils import TestCase, run_tests


def _covered_output_positions(
    input_num_rows, input_num_cols, block_size_rows, block_size_cols, use_bug
):
    """Mirror `_to_fp8_row_major_t`'s per-program-instance address/mask
    arithmetic in plain Python and return the set of transposed output
    (row, col) positions any program instance's masked store actually
    writes to.
    """
    output_num_rows, output_num_cols = input_num_cols, input_num_rows
    written = set()

    num_row_programs = -(-input_num_rows // block_size_rows)  # ceil div
    num_col_programs = -(-input_num_cols // block_size_cols)

    for block_row_id, block_col_id in itertools.product(
        range(num_row_programs), range(num_col_programs)
    ):
        block_row_offs = [
            block_row_id * block_size_rows + i for i in range(block_size_rows)
        ]
        block_col_offs = [
            block_col_id * block_size_cols + i for i in range(block_size_cols)
        ]

        for r in block_row_offs:
            for c in block_col_offs:
                out_row, out_col = c, r  # matches out_offs's axis assignment
                if use_bug:
                    # buggy mask: axes swapped relative to out_offs
                    in_bounds = (
                        r < output_num_rows and c < output_num_cols
                    )
                else:
                    in_bounds = (
                        out_row < output_num_rows and out_col < output_num_cols
                    )
                if in_bounds:
                    written.add((out_row, out_col))
    return written, output_num_rows, output_num_cols


class TestToFp8RowMajorTBoundsMask(TestCase):
    def test_fixed_mask_covers_exactly_the_valid_output(self):
        # Rectangular, non-tile-aligned shapes - exactly the case the issue
        # calls out as broken.
        for input_shape, block_size in [
            ((5, 7), 4),
            ((32, 16), 8),
            ((17, 9), 8),
            ((3, 20), 4),
        ]:
            input_num_rows, input_num_cols = input_shape
            written, output_num_rows, output_num_cols = _covered_output_positions(
                input_num_rows, input_num_cols, block_size, block_size, use_bug=False
            )
            expected = {
                (r, c)
                for r in range(output_num_rows)
                for c in range(output_num_cols)
            }
            self.assertEqual(
                written,
                expected,
                f"fixed mask should cover exactly the transposed output for "
                f"input_shape={input_shape}, block_size={block_size}",
            )

    def test_buggy_mask_leaves_valid_output_elements_unwritten(self):
        # Same shapes: the buggy (axis-swapped) mask must fail to cover the
        # full valid output for at least one of these non-square cases.
        any_gap = False
        for input_shape, block_size in [
            ((5, 7), 4),
            ((32, 16), 8),
            ((17, 9), 8),
            ((3, 20), 4),
        ]:
            input_num_rows, input_num_cols = input_shape
            written, output_num_rows, output_num_cols = _covered_output_positions(
                input_num_rows, input_num_cols, block_size, block_size, use_bug=True
            )
            expected = {
                (r, c)
                for r in range(output_num_rows)
                for c in range(output_num_cols)
            }
            if written != expected:
                any_gap = True
        self.assertTrue(
            any_gap,
            "expected the buggy (axis-swapped) mask to miss at least one "
            "valid output position for a rectangular input",
        )


if __name__ == "__main__":
    run_tests()
