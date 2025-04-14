
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


torch.manual_seed(32)

# class TestActivation24(common_utils.TestCase):

# @common_utils.parametrize("pattern", [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
def test_correctness():
    """
    Tests to see if the metadata packing format has changed between bf16 -> fp8, it looks like it's the same. 
    """
    # 238 in binary
    W_ref_asdf = torch.Tensor([0, 0, 1, 1]).to(device=device, dtype=dtype).tile((32, 64// 4)).contiguous()
    W_subclass_sparse = to_sparse_semi_structured(W_ref_asdf)

    garbanzo_beans = W_subclass_sparse.meta.view(torch.uint8).tolist()

    pattern = [1, 1, 0, 0] # 68
    for i in range(32):
        for j in range(8):
            W_ref = W_ref_asdf.clone()
            num_per_tb = 8
            W_ref[i, j*num_per_tb:(j+1)*num_per_tb] = torch.Tensor(pattern).to(device=device, dtype=dtype).tile((1, 2)).contiguous()

            # W_meta = to_sparse_semi_structured(W_ref).meta.view(torch.uint8)
            W_quant_func = _float8_cutlass_quant_sparse
            W_aqt = W_quant_func(W_ref, dtypeq_W)
            W_meta = W_aqt.tensor_impl.meta
            W_meta = W_meta[:32, :8]

            indicies = (W_meta == 68).nonzero()

            for (r, c) in indicies:
                garbanzo_beans[r][c] = f"a[{i:2d}, {j*num_per_tb:2d}:{(j+1)*num_per_tb:2d}]"

    # from pprint import pprint
    for line in garbanzo_beans:
        print(line[:4])
        print(line[4:])

    assert False
    # torch.testing.assert_close(W_meta, W_subclass_sparse.meta.view(torch.uint8))


def test_fast_rowwise_packing():
    # W_ref = create_semi_structured_tensor(128, 128)
    W_ref = create_semi_structured_tensor(128, 128, dtype=dtype).to(device)
    W_subclass_sparse = to_sparse_semi_structured(W_ref)
    # print(W_ref)


    # Test packed
    vc_mine = torch.unique(packed, return_counts=True)
    vc_ref = torch.unique(W_subclass_sparse.packed, return_counts=True)
    print(packed[:16, :16])
    print(W_subclass_sparse.packed[:16, :16])
    torch.testing.assert_close(vc_mine, vc_ref)

    # Test meta
    # vc_mine = torch.unique(packed_meta, return_counts=True)
    # vc_ref = torch.unique(W_subclass_sparse.meta, return_counts=True)
    # torch.testing.assert_close(vc_mine, vc_ref)


def test_packed_fp8():
    # W_ref = create_semi_structured_tensor(128, 128, dtype=torch.float8_e4m3fn).to(device)
    W_ref = torch.Tensor([[2, 3, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 8, 0, 0], 
                          [0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8], 
                          [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0],
                          [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]]).to(device=device).tile((128// 4, 128// 16)).contiguous().to(torch.float8_e4m3fn)
    packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)
    packed, packed_meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)
    
    torch.testing.assert_close(packed.to(torch.float16), packed_reference.to(torch.float16))


def test_meta_fp8_fixed():
    torch.manual_seed(123)
    W_ref = create_semi_structured_tensor(128, 128, dtype=torch.float8_e4m3fn).to(device)
    # W_ref = torch.Tensor([[2, 3, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 8, 0, 8, 0], 
    #                       [0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8], 
    #                       [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0],
    #                       [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]]).to(device=device).tile((128// 4, 128// 16)).contiguous().to(torch.float8_e4m3fn)
    packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)
    packed, packed_meta = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)

    vc_mine = torch.unique(packed_meta, return_counts=True)
    vc_ref = torch.unique(meta_reference, return_counts=True)
    # print(vc_mine)
    # print(packed_meta[:16, :16])
    # print(meta_reference[:16, :16])

    # print(packed_meta - meta_reference)
    # torch.testing.assert_close(packed, packed_reference)
    torch.testing.assert_close(packed_meta, meta_reference)


# common_utils.instantiate_parametrized_tests(TestActivation24)
# 

    # pprint(garbanzo_beans)



    # print(W_meta)

    # breakpoint()
    # print(W_subclass_sparse.meta.view(torch.uint8) == W_meta)
    # print("CUTLASS REFERENCE")
    # print(W_meta)
    # print(W_meta.shape)
    # print(packed_meta)
    # packed, packed_meta, packed_t, packed_t_meta , bitmask = torch.ops.torchao.sparse_semi_structured_tile.default(W_ref, "", True)
    # print(W_meta)
