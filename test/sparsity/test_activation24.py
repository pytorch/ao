
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
    r, c = 128, 256
    # 238 in binary
    W_ref_asdf = torch.Tensor([0, 0, 1, 1]).to(device=device, dtype=torch.float8_e4m3fn).tile((r, c // 4)).contiguous()
    packed_reference, meta_reference = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref_asdf)
    # W_quant_func = _float8_cutlass_quant_sparse
    # W_aqt = W_quant_func(W_ref_asdf, dtypeq_W)
    # W_meta = W_aqt.tensor_impl.meta
    print("INITIAL")
    print(meta_reference)
    print(meta_reference.shape, meta_reference.is_contiguous(), meta_reference.dtype)
    breakpoint()
    garbanzo_beans = meta_reference.tolist()


    pattern = [1, 1, 0, 0] # 68
    for i in range(r):
        num_per_tb = 8
        for j in range(c // num_per_tb):
            W_ref = W_ref_asdf.clone()
            W_ref[i, j*num_per_tb:(j+1)*num_per_tb] = torch.Tensor(pattern).to(device=device, dtype=dtype).tile((1, 2)).contiguous()
            _, W_meta = to_sparse_semi_structured_cutlass_sm9x_f8(W_ref)

            indicies = (W_meta == 68).nonzero()

            # print(indicies, i, j, W_meta)
            # breakpoint()

            for (r_i, c_i) in indicies:
                garbanzo_beans[r_i][c_i] = f"a[{i:2d}, {j*num_per_tb:2d}:{(j+1)*num_per_tb:2d}]"

    # from pprint import pprint
    print("METADATA FORMAT")
    for line in garbanzo_beans:
        print(line)
        print()
        # print(line[:4])
        # print(line[4:])

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
