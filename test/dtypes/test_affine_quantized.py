from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
from torchao.quantization import (
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    float8_weight_only,
)
from torchao.dtypes import SemiSparseLayout
from torch.testing._internal import common_utils
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

import torch
import unittest
import tempfile

is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def get_quantization_functions(do_sparse: bool, do_int4: bool):
    base_functions = [
        int8_weight_only(),
        int8_dynamic_activation_int4_weight(),
        int8_dynamic_activation_int8_weight(),
    ]
    if do_int4:
        base_functions.append(int4_weight_only(group_size=32))

    if do_sparse:
        base_functions.append(int8_dynamic_activation_int8_weight(layout=SemiSparseLayout()))

    if is_cuda_8_9:
        base_functions.append(float8_weight_only())

    return base_functions


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
    @common_utils.parametrize("apply_quant", get_quantization_functions(True, True))
    def test_weights_only(self, apply_quant):
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
    @common_utils.parametrize("apply_quant", get_quantization_functions(False, False))
    def test_to_device(self, apply_quant):
        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        ql = apply_quant(l)
        ql.to("cuda")

        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        ql = apply_quant(l)
        ql.to(device="cuda")

        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        ql = apply_quant(l)
        ql.cuda()

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_register_new_dispatch(self):
        from torchao.dtypes.affine_quantized_tensor import (
            register_aqt_quantized_linear_dispatch,
            deregister_aqt_quantized_linear_dispatch,
        )
        from torchao.dtypes import to_affine_quantized_intx
        from torchao.dtypes import AffineQuantizedTensor
        from torchao.quantization.quant_primitives import MappingType

        def dispatch_condition(input_tensor, weight_tensor, bias):
            return (
                isinstance(weight_tensor, AffineQuantizedTensor) and
                weight_tensor.quant_min == 0 and
                weight_tensor.quant_max == 2**6-1
            )

        def impl(input_tensor, weight_tensor, bias):
            # this is just for testing, normally people will call into uint6 weight only
            # quantized linear operator here
            assert False, "dispatching to my impl for uint6 weight only quant"

        register_aqt_quantized_linear_dispatch(dispatch_condition, impl)

        def apply_uint6_weight_only_quant(linear):
            linear.weight = torch.nn.Parameter(to_affine_quantized_intx(linear.weight, MappingType.ASYMMETRIC, (1, linear.weight.shape[-1]), torch.uint8, 0, 2**6-1), requires_grad=False)
            return linear

        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        apply_uint6_weight_only_quant(l)

        example_input = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(AssertionError, "dispatching to my impl for uint6 weight only quant"):
            l(example_input)

        deregister_aqt_quantized_linear_dispatch(dispatch_condition)

    @common_utils.parametrize("apply_quant", get_quantization_functions(True, True))
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_print_quantized_module(self, apply_quant):
        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        ql = apply_quant(l)
        assert "AffineQuantizedTensor" in str(ql)


common_utils.instantiate_parametrized_tests(TestAffineQuantized)


if __name__ == "__main__":
    run_tests()
