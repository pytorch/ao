# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/4634.

Both Triton GEMMs in torchao/prototype/blockwise_fp8_training/kernels.py
(`triton_fp8_gemm_1x128_128x128_kernel` and
`triton_fp8_gemm_1x128_128x1_kernel`) load their per-tile `a_s`/`b_s` scale
values without a bounds mask, while every other load and the final store in
those kernels is masked. When the autotuner's tile does not evenly divide
`M` or `N`, the scale loads dereference memory past the end of `a_s`/`b_s`
for the tail tile - a real out-of-bounds read reported via
compute-sanitizer in the issue (the GEMM's numeric output happens to still
be correct, since `c_mask` drops those lanes at the store; this is a
memory-safety bug, not a numerical one).

These kernels require CUDA to run (they're `@triton.jit`-compiled GEMMs
targeting an H100/B200), so they can't be exercised in this environment.
Instead, this test isolates the exact `offs_m`/`offs_n` -> scale-index
arithmetic in plain Python for tile geometries where `M`/`N` are *not* a
multiple of the block size, and checks whether any program instance's scale
load index falls outside the valid `[0, M)`/`[0, N)` range - i.e. whether
the load the kernel issues is actually in-bounds, mirroring
`mask=offs_m < M` / `mask=offs_n < N`.
"""

import itertools

from torch.testing._internal.common_utils import TestCase, run_tests


def _scale_load_indices(M, N, block_size_m, block_size_n):
    """Return the set of row indices used for the `a_s` load (indexed by
    `offs_m`) and column indices used for the `b_s` load (indexed by
    `offs_n`), across every program instance the kernel launches - exactly
    the indices `tl.load(..., mask=offs_m < M)` / `mask=offs_n < N` would
    guard.
    """
    num_m_programs = -(-M // block_size_m)  # ceil div, matches tl.cdiv grid
    num_n_programs = -(-N // block_size_n)

    a_s_indices = set()
    b_s_indices = set()
    for pid_m, pid_n in itertools.product(range(num_m_programs), range(num_n_programs)):
        offs_m = [pid_m * block_size_m + i for i in range(block_size_m)]
        offs_n = [pid_n * block_size_n + i for i in range(block_size_n)]
        a_s_indices.update(offs_m)
        b_s_indices.update(offs_n)
    return a_s_indices, b_s_indices


class TestGemmScaleBounds(TestCase):
    def test_unmasked_scale_loads_go_out_of_bounds(self):
        # Shapes where M/N are not a whole number of block-size tiles -
        # exactly the case the issue calls out.
        cases = [
            (M, N, block) for (M, N, block) in [(96, 96, 64), (100, 130, 32), (65, 65, 64)]
        ]
        any_oob = False
        for M, N, block in cases:
            a_s_indices, b_s_indices = _scale_load_indices(M, N, block, block)
            a_s_oob = {i for i in a_s_indices if i >= M}
            b_s_oob = {i for i in b_s_indices if i >= N}
            if a_s_oob or b_s_oob:
                any_oob = True
        self.assertTrue(
            any_oob,
            "expected at least one (M, N, block_size) combination with M or N "
            "not a multiple of block_size to produce an out-of-range scale "
            "load index, matching the unmasked kernel's behavior",
        )

    def test_masked_scale_loads_never_go_out_of_bounds(self):
        # With mask=offs_m < M / mask=offs_n < N applied (as the fix adds),
        # every scale load index that is actually dereferenced (i.e. not
        # masked off) must be in [0, M) / [0, N).
        for M, N, block in [(96, 96, 64), (100, 130, 32), (65, 65, 64), (128, 128, 64)]:
            a_s_indices, b_s_indices = _scale_load_indices(M, N, block, block)
            masked_a_s = {i for i in a_s_indices if i < M}
            masked_b_s = {i for i in b_s_indices if i < N}
            self.assertTrue(all(0 <= i < M for i in masked_a_s))
            self.assertTrue(all(0 <= i < N for i in masked_b_s))
            # and the masked set still covers every valid row/col at least once
            self.assertEqual(masked_a_s, set(range(M)))
            self.assertEqual(masked_b_s, set(range(N)))


if __name__ == "__main__":
    run_tests()
