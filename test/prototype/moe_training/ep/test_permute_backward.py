# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for the non-Triton backward of `_PermuteMXFP8FwdHPBwd`.

The full forward/backward flow requires the Triton index-generation kernel
(CUDA), but the non-Triton backward branch only reads `ctx.saved_tensors`,
`ctx.padded_input_shape` and `ctx.use_triton_for_bwd`, so it can be exercised
on CPU by invoking the backward directly with a lightweight context.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.testing

# `permute.py` imports triton at module level, so importing the module under
# test requires triton even though this test only exercises its non-Triton
# branch. Skip (rather than error at collection time) on runners without it.
pytest.importorskip("triton")

from torchao.prototype.moe_training.ep.permute import _PermuteMXFP8FwdHPBwd


def test_backward_non_triton_drops_padding_row():
    """The non-Triton backward must drop the appended padding row.

    `forward` appends one dummy zero row to the input and maps `-1` entries of
    `permuted_indices` to it, so the backward must return exactly `input_rows - 1`
    rows — the gradients of the original (unpadded) input — matching the Triton
    backward branch.
    """
    M, K = 5, 4
    # valid indices first, then -1 dummies; every -1 maps to the padding row
    # (index M) via Python's negative indexing on the scatter target
    permuted_indices = torch.tensor([2, -1, 0, -1, 4, -1], dtype=torch.int64)
    grad_output = torch.randn(permuted_indices.numel(), K)

    ctx = SimpleNamespace(
        saved_tensors=(permuted_indices,),
        padded_input_shape=(M + 1, K),
        use_triton_for_bwd=False,
    )

    grad_input, *_ = _PermuteMXFP8FwdHPBwd.backward(
        ctx, None, grad_output, None, None, None
    )

    # regression guard: must NOT include the padding row (M+1 rows)
    assert grad_input.shape == (M, K), (
        f"expected {(M, K)} rows (original input), got {tuple(grad_input.shape)}; "
        "the padding row was not removed"
    )

    expected = torch.zeros(M, K)
    for row, idx in enumerate(permuted_indices.tolist()):
        if idx != -1:
            expected[idx] = grad_output[row]
    torch.testing.assert_close(grad_input, expected)
