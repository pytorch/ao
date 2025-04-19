
import torch
import torchao
import torch.nn.functional as F

from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8, sparse_semi_structured_tile
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
    _float8_cutlass_quant_sparse
)
torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True

from torchao.sparsity.utils import create_semi_structured_tensor

dtype = torch.float16
device = torch.device("cuda")
dtypeq_X = torch.float8_e4m3fn
dtypeq_W = torch.float8_e4m3fn
from torchao.utils import is_sm_at_least_90
import unittest


torch.manual_seed(32)

@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
@unittest.skip("Not implemented yet")
def test_sparse24_sm90_sparsify_fp8(
    M=512, K=1024, fp8=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A_sp_ref = create_semi_structured_tensor(M, K, dtype=torch.float8_e4m3fn).to(device)
    
    # Test with act="identity"
    A_packed_ref, A_mdata_ref = to_sparse_semi_structured_cutlass_sm9x_f8(
        A_sp_ref.to(fp8)
    )
    A_packed, A_mdata = torch.ops.torchao.sparse24_sm90_sparsify(
        A_sp_ref,
        "cutlass",
        "identity",
        sp_selection_algo="largest",
        dtype=A_packed_ref.dtype,
    )

    assert (A_packed != A_packed_ref).float().mean().item() < 0.005
    assert (A_mdata != A_mdata_ref).float().mean().item() < 0.005
    assert torch.allclose(A_packed.float().sum(), A_packed_ref.float().sum())

# @fairinternal-below
@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
def test_sparse24_sm90_sparsify_identity(
    M=512, K=1024, fp8=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A_sp_ref = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).to(device)
    
    # Test with act="identity"
    A_packed_ref, A_mdata_ref = to_sparse_semi_structured_cutlass_sm9x_f8(
        A_sp_ref.to(fp8)
    )
    A_packed, A_mdata = torch.ops.torchao.sparse24_sm90_sparsify(
        A_sp_ref,
        "cutlass",
        "identity",
        sp_selection_algo="largest",
        dtype=A_packed_ref.dtype,
    )

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
    A_dense = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).to(device)
    A_scale = torch.randn([M, 1], device="cuda", dtype=torch.float32).abs() + 0.1
    A_sp_ref =  (A_dense / A_scale).bfloat16()

    A_packed_ref, A_mdata_ref = to_sparse_semi_structured_cutlass_sm9x_f8(
        A_sp_ref.to(fp8)
    )
    A_packed, A_mdata = torch.ops.torchao.sparse24_sm90_sparsify(
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
    A_dense = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).to(device)
    A_sp_ref = (A_dense.float().relu() ** 2).bfloat16()

    # Test with act="srelu"
    # NOTE: Due to different rounding strategies, and way more zeros, we don't have the exact same
    # bitwise packed values, so we bump up the margin here
    A_packed_ref, _A_mdata_ref = to_sparse_semi_structured_cutlass_sm9x_f8(
        A_sp_ref.to(fp8)
    )
    A_packed, _A_mdata = torch.ops.torchao.sparse24_sm90_sparsify(
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
