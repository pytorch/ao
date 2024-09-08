import torch
import copy
import pytest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5
from torchao.dtypes import MarlinSparseLayoutType
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.quant_api import int4_weight_only, quantize_
from torchao.sparsity.marlin import (
    pack_to_marlin_24,
    unpack_from_marlin_24,
    inject_24
)
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    ZeroPointDomain,
    MappingType,
)


class SparseMarlin24(TestCase):

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

        self.input = torch.randn((32, 16, 4096), dtype=torch.float16, device="cuda")
        self.model = (
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_quant_sparse_marlin_layout_eager(self):
        apply_fake_sparsity(self.model)
        model_copy = copy.deepcopy(self.model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(self.input.bfloat16()).half()

        # Sparse + quantized
        quantize_(self.model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = self.model(self.input)

        assert torch.allclose(dense_result, sparse_result, atol=3e-1), "Results are not close"

    @pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_5, reason="Needs PyTorch 2.5+")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_quant_sparse_marlin_layout_compile(self):
        apply_fake_sparsity(self.model)
        model_copy = copy.deepcopy(self.model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        model_copy.foward = torch.compile(model_copy.forward, fullgraph=True)
        dense_result = model_copy(self.input.bfloat16()).half()

        # Sparse + quantized
        quantize_(self.model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        self.model.forward = torch.compile(self.model.forward, fullgraph=True)
        sparse_result = self.model(self.input)

        assert torch.allclose(dense_result, sparse_result, atol=3e-1), "Results are not close"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_pack_unpack_equivalence(self):
        num_bits = 4
        group_size = 128
        shape = (11008, 4096)
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        zero_point_dtype = torch.bfloat16
        mapping_type = MappingType.SYMMETRIC
        preserve_zero = True
        zero_point_domain = ZeroPointDomain.INT
        scale_dtype = None

        w = torch.rand(shape, dtype=torch.float16, device="cuda")

        # Inject 2:4 sparsity mask
        w_24, _ = inject_24(w, *w.shape)

        # Quantize weights 
        scales, zeros = choose_qparams_affine(w_24, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
        w_q_24 = quantize_affine(w_24, block_size, scales, zeros, target_dtype, quant_min, quant_max, zero_point_domain)
        scales = scales.reshape(-1, w_q_24.shape[1])

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
