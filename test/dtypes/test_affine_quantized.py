import tempfile
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.dtypes import Int4CPULayout, SemiSparseLayout
from torchao.quantization import (
    float8_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    is_sm_at_least_89,
)


def get_quantization_functions(do_sparse: bool, do_int4: bool, device: str = "cuda"):
    base_functions = [
        int8_weight_only(),
        int8_dynamic_activation_int4_weight(),
        int8_dynamic_activation_int8_weight(),
        int8_dynamic_activation_int8_weight(act_mapping_type=MappingType.ASYMMETRIC),
    ]
    if do_int4:
        if device == "cpu" and TORCH_VERSION_AT_LEAST_2_6:
            base_functions.append(
                int4_weight_only(group_size=32, layout=Int4CPULayout())
            )
        else:
            base_functions.append(int4_weight_only(group_size=32))

    if do_sparse:
        base_functions.append(
            int8_dynamic_activation_int8_weight(layout=SemiSparseLayout())
        )

    if is_sm_at_least_89():
        base_functions.append(float8_weight_only())

    return base_functions


class TestAffineQuantized(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tensor_core_layout_transpose(self):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        t = linear.weight
        shape = t.shape
        apply_int4_weight_only_quant = int4_weight_only(group_size=32)
        ql = apply_int4_weight_only_quant(linear)
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
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        ql = apply_quant(linear)
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
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        ql = apply_quant(linear)
        ql.to("cuda")

        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        ql = apply_quant(linear)
        ql.to(device="cuda")

        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        ql = apply_quant(linear)
        ql.cuda()

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_register_new_dispatch(self):
        from torchao.dtypes import AffineQuantizedTensor, to_affine_quantized_intx
        from torchao.dtypes.affine_quantized_tensor_ops import (
            deregister_aqt_quantized_linear_dispatch,
            register_aqt_quantized_linear_dispatch,
        )
        from torchao.quantization.quant_primitives import MappingType

        def dispatch_condition(input_tensor, weight_tensor, bias):
            return (
                isinstance(weight_tensor, AffineQuantizedTensor)
                and weight_tensor.quant_min == 0
                and weight_tensor.quant_max == 2**6 - 1
            )

        def impl(input_tensor, weight_tensor, bias):
            # this is just for testing, normally people will call into uint6 weight only
            # quantized linear operator here
            assert False, "dispatching to my impl for uint6 weight only quant"

        register_aqt_quantized_linear_dispatch(dispatch_condition, impl)

        def apply_uint6_weight_only_quant(linear):
            linear.weight = torch.nn.Parameter(
                to_affine_quantized_intx(
                    linear.weight,
                    MappingType.ASYMMETRIC,
                    (1, linear.weight.shape[-1]),
                    torch.uint8,
                    0,
                    2**6 - 1,
                ),
                requires_grad=False,
            )
            return linear

        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        apply_uint6_weight_only_quant(linear)

        example_input = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(
            AssertionError, "dispatching to my impl for uint6 weight only quant"
        ):
            linear(example_input)

        deregister_aqt_quantized_linear_dispatch(dispatch_condition)

    @common_utils.parametrize("apply_quant", get_quantization_functions(True, True))
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_print_quantized_module(self, apply_quant):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        ql = apply_quant(linear)
        assert "AffineQuantizedTensor" in str(ql)


class TestAffineQuantizedBasic(TestCase):
    COMMON_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    COMMON_DTYPES = [torch.bfloat16]

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_flatten_unflatten(self, device, dtype):
        apply_quant_list = get_quantization_functions(False, True, device)
        for apply_quant in apply_quant_list:
            linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
            ql = apply_quant(linear)
            lp_tensor = ql.weight
            tensor_data_name_dict, tensor_attributes = lp_tensor.__tensor_flatten__()
            tensor_data_dict = {
                name: getattr(lp_tensor, name) for name in tensor_data_name_dict
            }
            outer_size = lp_tensor.size()
            outer_stride = lp_tensor.stride()
            reconstructed = type(lp_tensor).__tensor_unflatten__(
                tensor_data_dict, tensor_attributes, outer_size, outer_stride
            )
            example_inputs = (torch.randn(32, 128, dtype=dtype, device=device),)
            ref = ql(*example_inputs)
            ql.weight = torch.nn.Parameter(reconstructed, requires_grad=False)
            reconstruct_res = ql(*example_inputs)
            self.assertEqual(reconstruct_res, ref)


common_utils.instantiate_parametrized_tests(TestAffineQuantized)
common_utils.instantiate_parametrized_tests(TestAffineQuantizedBasic)


if __name__ == "__main__":
    run_tests()
