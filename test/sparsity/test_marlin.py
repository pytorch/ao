import torch
import copy
import pytest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.dtypes import MarlinSparseLayoutType
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.quant_api import int4_weight_only, quantize_
from torchao.sparsity.marlin import (
    pack_to_marlin_24,
    unpack_from_marlin_24,
    inject_24
)
from torchao.quantization.utils import (
    get_group_qparams_symmetric,
    groupwise_affine_quantize_tensor_from_qparams,
)


class SparseMarlin24(TestCase):

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_quant_sparse_marlin_layout_eager(self):
        torch.manual_seed(0)

        input = torch.randn((32, 16, 4096), dtype=torch.float16, device="cuda")
        model = (
            nn.Sequential(
                nn.Linear(4096, 21504),
                nn.Linear(21504, 4096),
                nn.ReLU(),
                nn.Linear(4096, 21504),
                nn.Linear(21504, 4096),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, atol=3e-1), "Results are not close"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_quant_sparse_marlin_layout_compile(self):
        torch.manual_seed(0)

        input = torch.randn((32, 16, 4096), dtype=torch.float16, device="cuda")
        model = (
            nn.Sequential(
                nn.Linear(4096, 21504),
                nn.Linear(21504, 4096),
                nn.ReLU(),
                nn.Linear(4096, 21504),
                nn.Linear(21504, 4096),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        model_copy.foward = torch.compile(model_copy.forward, fullgraph=True)
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        model.forward = torch.compile(model.forward, fullgraph=True)
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, atol=3e-1), "Results are not close"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_pack_unpack_equivalence(self):
        num_bits = 4
        group_size = 128
        shape = (11008, 4096)
        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        w = torch.rand(shape, dtype=torch.float16, device="cuda")
        size_k, size_n = w.shape

        # Inject 2:4 sparsity mask
        w_24, _ = inject_24(w, *w.shape)

        # Quantize weights 
        w_24 = w_24.reshape((-1, group_size, size_n))
        w_24 = w_24.permute(1, 0, 2)
        w_24 = w_24.reshape((group_size, -1))

        # Compute scale for each group
        scales = torch.max(torch.abs(w_24), 0, keepdim=True)[0]
        scales *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        w_q_24 = torch.round(w_24 / scales).int()
        w_q_24 += half_q_val
        w_q_24 = torch.clamp(w_q_24, 0, max_q_val)

        # Shape back to original shape
        w_q_24 = w_q_24.reshape((group_size, -1, size_n))
        w_q_24 = w_q_24.permute(1, 0, 2)
        w_q_24 = w_q_24.reshape((size_k, size_n)).contiguous()
        scales = scales.reshape((-1, size_n)).contiguous()

        # Test pack/unpack equivalence
        q_w_comp, packed_scales, meta = pack_to_marlin_24(
            w_q_24, scales, num_bits, group_size
        )
        unpacked_q_w, unpacked_scales = unpack_from_marlin_24(
            q_w_comp, packed_scales, meta, shape, group_size, num_bits
        )

        assert torch.equal(w_q_24, unpacked_q_w), "Unpacked weights do not match original weights"
        assert torch.equal(scales, unpacked_scales), "Unpacked scales do not match original scales"


if __name__ == "__main__":
    run_tests()
