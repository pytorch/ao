import itertools

import bitsandbytes.functional as F
import pytest
import torch

from torchao.prototype.galore.kernels import triton_quantize_blockwise

SEED = 0
torch.manual_seed(SEED)

DIM1 = [64, 1024, 4096]
DIM2 = [1024, 2048, 4096]
SIGNS = [True, False]
DTYPES = [torch.float32]  # , torch.float16]
BLOCKSIZE = [2048]

QUANT_CONFIG = list(itertools.product(DIM1, DIM2, DTYPES, SIGNS, BLOCKSIZE))


@pytest.mark.parametrize(
    "dim1,dim2,dtype,signed,blocksize",
    QUANT_CONFIG,
)
def test_quantize_blockwise(dim1, dim2, dtype, signed, blocksize):
    g = torch.randn(dim1, dim2, device="cuda", dtype=dtype) * 0.01

    qmap = F.create_dynamic_map(signed).to(g.device)

    ref_bnb, qstate = F.quantize_blockwise(g, code=qmap, blocksize=blocksize)
    bnb_norm = (g.reshape(-1, blocksize) / qstate.absmax[:, None]).reshape(g.shape)

    tt_q, tt_norm, tt_absmax = triton_quantize_blockwise(
        g, qmap, group_size=blocksize, return_normalized=True
    )
    tt_check = torch.allclose(ref_bnb, tt_q)
    if not tt_check:
        print(
            f"Failed quantization check for {dim1} x {dim2}, {dtype}, signed {signed}"
        )
        print(f"Absmax: {(qstate.absmax - tt_absmax).abs().max()}")
        print(f"Norm diff: {(bnb_norm - tt_norm).abs().max()}")

        idx_tt = (ref_bnb != tt_q).to("cuda")
        print(f"Num diffs vs bnb: {idx_tt.sum()}")
        max_idx_diff = (ref_bnb - tt_q).abs().max()
        print(f"Max idx diff vs bnb: {max_idx_diff}")

    assert tt_check or (not tt_check and max_idx_diff <= 1)


# if __name__ == "__main__":
#     for d1, d2, dtype, signed, blocksize in QUANT_CONFIG:
#         test_quantize_blockwise(d1, d2, dtype, signed, blocksize)
