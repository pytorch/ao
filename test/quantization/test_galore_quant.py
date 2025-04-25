# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools

import pytest

from torchao.utils import TORCH_VERSION_AT_LEAST_2_7

# Skip entire test if triton is not available, otherwise CI failure
try:  # noqa: F401
    import triton  # noqa: F401
except ImportError:  # noqa: F401
    pytest.skip("triton is not installed", allow_module_level=True)  # noqa: F401
import torch

# Skip entire test if CUDA is not available or ROCM is enabled
if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip(
        "CUDA is not available/ ROCM support is under development",
        allow_module_level=True,
    )

from bitsandbytes.functional import (
    create_dynamic_map,
    dequantize_blockwise,
    quantize_blockwise,
)

from torchao.prototype.galore.kernels import (
    triton_dequant_blockwise,
    triton_quantize_blockwise,
)
from torchao.testing.utils import skip_if_rocm

SEED = 0
torch.manual_seed(SEED)

DIM1 = [64, 1024, 4096]
DIM2 = [1024, 2048, 4096]
SIGNS = [True, False]
DTYPES = [torch.float32]  # , torch.float16]
BLOCKSIZE = [2048]

TEST_CONFIGS = list(itertools.product(DIM1, DIM2, DTYPES, SIGNS, BLOCKSIZE))


@pytest.mark.skip("skipping for now, see comments below")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
@pytest.mark.parametrize(
    "dim1,dim2,dtype,signed,blocksize",
    TEST_CONFIGS,
)
def test_galore_quantize_blockwise(dim1, dim2, dtype, signed, blocksize):
    g = torch.randn(dim1, dim2, device="cuda", dtype=dtype) * 0.01

    qmap = create_dynamic_map(signed).to(g.device)

    ref_bnb, qstate = quantize_blockwise(g, code=qmap, blocksize=blocksize)
    bnb_norm = (g.reshape(-1, blocksize) / qstate.absmax[:, None]).reshape(g.shape)

    tt_q, tt_norm, tt_absmax = triton_quantize_blockwise(
        g, qmap, group_size=blocksize, return_normalized=True
    )
    tt_check = torch.allclose(ref_bnb, tt_q)

    # see notes.md under `prototype.galore.kernels` for an explanation of the following conditions
    if not tt_check:
        print(
            f"Failed quantization check for {dim1} x {dim2}, {dtype}, signed {signed}"
        )
        print(f"Absmax: {(qstate.absmax - tt_absmax).abs().max()}")
        print(f"Norm diff: {(bnb_norm - tt_norm).abs().max()}")

        idx_diff = (ref_bnb != tt_q).to("cuda")
        print(f"Num code idx diffs: {idx_diff.sum()}")
        max_idx_diff = (ref_bnb - tt_q).abs().max()
        print(f"Max code idx diff: {max_idx_diff}")

        # This below checks that the value being quantized falls half-way between two code buckets
        # where bitsandbytes assigns to one and the triton implementation assigns to the other
        # Since either bucket is technically valid, we only check that the distance between the value and the
        # adjacent buckets are the same.  I.e., we don't require that the triton implementation exactly matches
        # bitsandbytes.

        bnb_code = qmap[ref_bnb[idx_diff].tolist()]
        tt_code = qmap[tt_q[idx_diff].tolist()]
        bnb_dist = torch.abs(bnb_code - bnb_norm[idx_diff])
        torch_dist = torch.abs(tt_code - bnb_norm[idx_diff])

        dist_sum = torch.sum(bnb_dist - torch_dist)
        print(f"Distance sum: {torch.sum(bnb_dist - torch_dist)}")
    assert tt_check or (not tt_check and dist_sum < 1e-4)


@pytest.mark.parametrize(
    "dim1,dim2,dtype,signed,blocksize",
    TEST_CONFIGS,
)
@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
@pytest.mark.skipif(
    TORCH_VERSION_AT_LEAST_2_7, reason="Failing in CI"
)  # TODO: fix this
def test_galore_dequant_blockwise(dim1, dim2, dtype, signed, blocksize):
    g = torch.randn(dim1, dim2, device="cuda", dtype=dtype) * 0.01

    qmap = create_dynamic_map(signed).to(g.device)

    q, qstate = quantize_blockwise(g, code=qmap, blocksize=blocksize)

    dq_ref = dequantize_blockwise(q, qstate)
    dq = triton_dequant_blockwise(q, qmap, qstate.absmax, group_size=blocksize)
    assert torch.allclose(dq, dq_ref)
