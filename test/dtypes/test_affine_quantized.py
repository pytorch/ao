from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from torchao.quantization.quant_api import (
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    float8_weight_only,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

import torch
import unittest
import tempfile


class TestAffineQuantized(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tensor_core_layout_transpose(self):
        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        t = l.weight
        shape = t.shape
        apply_int4_weight_only_quant = int4_weight_only(group_size=32)
        ql = apply_int4_weight_only_quant(l)
        aqt = ql.weight
        aqt_shape = aqt.shape
        self.assertEqual(aqt_shape, shape)

        # transpose shape test
        for _ in range(10):
            t = t.t()
            aqt = aqt.t()
            shape = t.shape
            aqt_shape = aqt.shape
            self.assertEqual(aqt_shape, shape)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_weights_only(self):
        for apply_quant in [int4_weight_only(group_size=32), int8_weight_only(), int8_dynamic_activation_int4_weight(),
                            int8_dynamic_activation_int8_weight(), int8_dynamic_activation_int8_semi_sparse_weight(), float8_weight_only()]:
            l = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
            ql = apply_quant(l)
            with tempfile.NamedTemporaryFile() as f:
                torch.save(ql.state_dict(), f)
                f.seek(0)
                # `weights_only=True` is enabled for torch 2.5+
                if TORCH_VERSION_AT_LEAST_2_5:
                    _ = torch.load(f, weights_only=True)
                else:
                    _ = torch.load(f, weights_only=False)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_to_device(self):
        from torchao.quantization import quantize_
        for apply_quant in [int8_weight_only(), int8_dynamic_activation_int4_weight(), int8_dynamic_activation_int8_weight()]:
            l = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            ql = apply_quant(l)
            ql.to("cuda")

            l = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            ql = apply_quant(l)
            ql.to(device="cuda")

            l = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            ql = apply_quant(l)
            ql.cuda()


if __name__ == "__main__":
    run_tests()
