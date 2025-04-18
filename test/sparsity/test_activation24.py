
import torch
import torchao
import torch.nn.functional as F

from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
    _float8_cutlass_quant_sparse
)
torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True

from torchao.sparsity.utils import create_semi_structured_tensor
from torch.sparse import to_sparse_semi_structured

from torch.testing._internal import common_utils

dtype = torch.float16
device = torch.device("cuda")
dtypeq_X = torch.float8_e4m3fn
dtypeq_W = torch.float8_e4m3fn
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=10000)
from torchao.testing.utils import skip_if_compute_capability_less_than
from torchao.utils import is_sm_at_least_90
import unittest


torch.manual_seed(32)

def test_packed_fp8():
    # W_ref = create_semi_structured_tensor(128, 128, dtype=torch.float8_e4m3fn).to(device)
    W_ref = torch.Tensor([[2, 3, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 8, 0, 0], 
                          [0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8], 
                          [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0],
                          [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]]).to(device=device).tile((128// 4, 128// 16)).contiguous().to(torch.float8_e4m3fn)
    packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)
    packed, packed_meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)
    
    torch.testing.assert_close(packed.to(torch.float16), packed_reference.to(torch.float16))


def test_meta_fp8_fixed_128x256():
    r, c = 128, 256
    torch.manual_seed(123)
    # W_ref = create_semi_structured_tensor(128, 256, dtype=torch.float8_e4m3fn).to(device)
    # print(W_ref[:18])
    # print(W_ref.count_nonzero())
    # print(W_ref)
    W_ref = torch.Tensor([[2, 3, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 8, 0, 8, 0], 
                          [0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8], 
                          [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0],
                          [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]]).to(device=device).tile((r // 4, c// 16)).contiguous().to(torch.float8_e4m3fn)
    packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)
    packed, packed_meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)

    # vc_mine = torch.unique(packed_meta, return_counts=True)
    # vc_ref = torch.unique(meta_reference, return_counts=True)
    # # print(vc_mine)
    print("CUSTOM")
    print(packed_meta[:16, :32])
    print("REFERENCE")
    print(meta_reference[:16, :32])

    # # print(packed_meta - meta_reference)
    torch.testing.assert_close(packed, packed_reference)
    torch.testing.assert_close(packed_meta, meta_reference)

def test_meta_packed_fp8():
    for r in (64, 128, 256, 512):
        for c in (128, 256, 512, 1024, 2048):
            torch.manual_seed(123)
            # random tensor without 0
            W_ref = create_semi_structured_tensor(r, c, dtype=torch.float8_e4m3fn).to(device)
            # W_ref = torch.Tensor([[2, 3, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 8, 0, 8, 0], 
            #                       [0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8], 
            #                       [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0],
            #                       [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]]).to(device=device).tile((r // 4, c// 16)).contiguous().to(torch.float8_e4m3fn)
            packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)
            packed, packed_meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)

            torch.testing.assert_close(packed, packed_reference)
            torch.testing.assert_close(packed_meta, meta_reference)

# @fairinternal-below
@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
def test_sparse24_sm90_sparsify_identity_1(
    M=512, K=1024, fp8=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A_sp_ref = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).to(device)
    
    # Test with act="identity"
    A_packed_ref, A_mdata_ref = to_sparse_semi_structured_cutlass_sm9x_f8(
        A_sp_ref.to(fp8)
    )
    A_packed, A_mdata = torch.ops.xformers.sparse24_sm90_sparsify(
        A_sp_ref,
        "cutlass",
        "identity",
        sp_selection_algo="largest",
        dtype=A_packed_ref.dtype,
    )
    A_mdata = A_mdata.view(A_mdata_ref.shape)

    # Note: sparsification is not deterministic (eg if 3 items have the same value in a block of 4 for instance)
    # so we allow a tiny margin for error
    assert (A_packed != A_packed_ref).float().mean().item() < 0.005
    assert (A_mdata != A_mdata_ref).float().mean().item() < 0.005
    # The sum should always match though
    assert torch.allclose(A_packed.float().sum(), A_packed_ref.float().sum())


@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
def test_sparse24_sm90_sparsify_identity_scaled(
    M=512, K=1024, fp8=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A_dense = torch.randn([M, K], device="cuda", dtype=torch.bfloat16)
    A_scale = torch.randn([M, 1], device="cuda", dtype=torch.float32).abs() + 0.1
    A_dense[A_dense == 0] = 1
    A_sp_ref = torch.ops.xformers.sparseNM_dense(
        (A_dense / A_scale).bfloat16(), N=2, M=4, sort_preproc="largest"
    )

    A_packed_ref, A_mdata_ref = torch.ops.xformers._sparse24_sm90_cutlass_compress(
        A_sp_ref.to(fp8)
    )
    A_packed, A_mdata = torch.ops.xformers.sparse24_sm90_sparsify(
        A_dense,
        "cutlass",
        "identity",
        sp_selection_algo="largest",
        dtype=A_packed_ref.dtype,
        scale=A_scale,
    )
    assert (A_packed != A_packed_ref).float().mean().item() < 0.05
    assert (A_mdata != A_mdata_ref).float().mean().item() < 0.005
    assert torch.allclose(
        A_packed.float().sum(), A_packed_ref.float().sum(), rtol=0.001
    )


@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
def test_sparse24_sm90_sparsify_srelu(M=512, K=1024, fp8=torch.float8_e4m3fn) -> None:
    torch.manual_seed(0)
    A_dense = torch.randn([M, K], device="cuda", dtype=torch.bfloat16)
    A_dense[A_dense == 0] = 1
    A_sp_ref = torch.ops.xformers.sparseNM_dense(
        (A_dense.float().relu() ** 2).bfloat16(), N=2, M=4, sort_preproc="largest"
    )

    # Test with act="srelu"
    # NOTE: Due to different rounding strategies, and way more zeros, we don't have the exact same
    # bitwise packed values, so we bump up the margin here
    A_packed_ref, _A_mdata_ref = torch.ops.xformers._sparse24_sm90_cutlass_compress(
        A_sp_ref.to(fp8)
    )
    A_packed, _A_mdata = torch.ops.xformers.sparse24_sm90_sparsify(
        A_dense,
        "cutlass",
        "srelu",
        sp_selection_algo="largest",
        dtype=A_packed_ref.dtype,
    )
    assert torch.allclose(
        A_packed.float().sum(), A_packed_ref.float().sum(), rtol=0.005
    )
    assert (A_packed != A_packed_ref).float().mean().item() < 0.1
