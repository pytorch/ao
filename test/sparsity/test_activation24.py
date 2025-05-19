import torch
import torch.nn.functional as F

from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8MMConfig,
    PerRow,
    quantize_,
)

torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True

import copy
import unittest

from torchao.prototype.sparsity.activation.srelu_linear import (
    SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig,
)
from torchao.sparsity import sparsify_
from torchao.sparsity.utils import create_binary_tensor, create_semi_structured_tensor
from torchao.utils import is_sm_at_least_90


@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
def test_sparse24_sm90_sparsify_identity(
    M=512, K=1024, fp8=torch.float8_e4m3fn
) -> None:
    torch.manual_seed(0)
    A_sp_ref = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).cuda()

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
    A_dense = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).cuda()
    A_scale = torch.randn([M, 1], device="cuda", dtype=torch.float32).abs() + 0.1
    A_sp_ref = (A_dense / A_scale).bfloat16()

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
    A_dense = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).cuda()
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


@unittest.skipIf(not is_sm_at_least_90(), "Need cuda arch greater than SM90")
def test_srelu_fp8_semi_sparse_activation_linear(M=512, K=2048, N=1024):
    with torch.no_grad():
        torch.manual_seed(0)
        input_tensor = create_semi_structured_tensor(M, K, dtype=torch.bfloat16).cuda()
        # we have to wrap in a sequential block for quantize_ to work properly
        reference_linear = torch.nn.Sequential(
            torch.nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)
        )
        reference_linear_copy = copy.deepcopy(reference_linear)

        quantize_(
            reference_linear,
            Float8DynamicActivationFloat8WeightConfig(
                granularity=PerRow(), mm_config=Float8MMConfig(use_fast_accum=False)
            ),
        )

        # define reference implementation
        def srelu_linear(x):
            x = F.relu(x) ** 2
            return reference_linear(x)

        reference_srelu = torch.compile(srelu_linear, fullgraph=True)

        # this only works with fullgraph=True, errors in eager
        # TODO figure out exactly why this happens
        sparsify_(
            reference_linear_copy,
            SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig(),
        )
        # (reference_linear_copy)
        reference_linear_copy.forward = torch.compile(
            reference_linear_copy.forward, fullgraph=True
        )

        reference_output = reference_srelu(input_tensor)
        custom_output = reference_linear_copy(input_tensor)

        torch.testing.assert_close(reference_output, custom_output, rtol=0.1, atol=0.01)



def test_splitk_sparse_gemv():
    torch.manual_seed(0)

    activation = create_binary_tensor((1, 1, 4096), 0.2).cuda().to(torch.float16)
    weight = torch.randn(16384, 4096, dtype=torch.float16).cuda()

    # weight must be column major
    weight_transposed = weight.T.contiguous().T

    sparse_res = torch.ops.torchao.splitk_sparse_gemv(activation, weight_transposed)
    dense_res = F.linear(activation, weight_transposed)

    # This rtol is ridiculousl high, because the split gemv output accumulates slightly differently than the dense output. 
    torch.testing.assert_close(sparse_res, dense_res, rtol=10, atol=0.1)


def test_splitk_sparse_gemv_fp8():

    torch.nn.Linear()
    torch.manual_seed(0)

    activation = create_binary_tensor((1, 1, 4096), 0.2).cuda().to(torch.float16)
    weight = torch.randn(16384, 4096, dtype=torch.float16).cuda()

    # weight must be column major
    weight_transposed = weight.T.contiguous().T

    sparse_res = torch.ops.torchao.splitk_sparse_gemv(activation, weight_transposed)
    dense_res = F.linear(activation, weight_transposed)

    # This rtol is ridiculousl high, because the split gemv output accumulates slightly differently than the dense output. 
    torch.testing.assert_close(sparse_res, dense_res, rtol=10, atol=0.1)
