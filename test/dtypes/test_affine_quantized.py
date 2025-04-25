# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    CutlassInt4PackedLayout,
    Int4CPULayout,
    Int4XPULayout,
    PlainLayout,
    SemiSparseLayout,
    to_affine_quantized_intx,
    to_affine_quantized_intx_static,
)
from torchao.quantization import (
    GemliteUIntXWeightOnlyConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    float8_weight_only,
    int4_dynamic_activation_int4_weight,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
)
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.testing.utils import skip_if_no_cuda, skip_if_no_gemlite, skip_if_rocm
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    check_cpu_version,
    check_xpu_version,
    is_fbcode,
    is_ROCM,
    is_sm_at_least_89,
)

is_cusparselt_available = (
    hasattr(torch.backends, "cusparselt") and torch.backends.cusparselt.is_available()
)


def get_quantization_functions(
    do_sparse: bool, do_int4: bool, device: str = "cuda", int4_zp_int: bool = False
):
    base_functions = [
        int8_weight_only(),
        int8_dynamic_activation_int4_weight(),
        int8_dynamic_activation_int8_weight(),
        int8_dynamic_activation_int8_weight(act_mapping_type=MappingType.ASYMMETRIC),
    ]
    if do_int4:
        if check_cpu_version(device):
            base_functions.append(
                int4_weight_only(group_size=32, layout=Int4CPULayout())
            )
        elif check_xpu_version(device):
            base_functions.append(
                int4_weight_only(group_size=32, layout=Int4XPULayout())
            )
            if int4_zp_int:
                base_functions.append(
                    int4_weight_only(
                        group_size=32,
                        layout=Int4XPULayout(),
                        zero_point_domain=ZeroPointDomain.INT,
                    )
                )
        else:
            base_functions.append(int4_weight_only(group_size=32))
            if device == "cuda" and not is_ROCM():
                base_functions.append(
                    int8_dynamic_activation_int4_weight(
                        group_size=None,
                        mapping_type=MappingType.SYMMETRIC,
                        act_mapping_type=MappingType.SYMMETRIC,
                        layout=CutlassInt4PackedLayout(),
                    )
                )
                base_functions.append(int4_dynamic_activation_int4_weight())

    if do_sparse and device != "xpu":
        base_functions.append(
            int8_dynamic_activation_int8_weight(layout=SemiSparseLayout())
        )

    if is_sm_at_least_89():
        base_functions.append(float8_weight_only())

    return base_functions


class TestAffineQuantized(TestCase):
    GPU_DEVICES = (["cuda"] if torch.cuda.is_available() else []) + (
        ["xpu"] if torch.xpu.is_available() else []
    )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tensor_core_layout_transpose(self):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        t = linear.weight
        shape = t.shape
        apply_int4_weight_only_quant = int4_weight_only(group_size=32)
        quantize_(linear, apply_int4_weight_only_quant)
        ql = linear
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

    @unittest.skipIf(len(GPU_DEVICES) == 0, "Need GPU available")
    def test_weights_only(self):
        for device in self.GPU_DEVICES:
            apply_quant_list = get_quantization_functions(
                is_cusparselt_available, True, device, True
            )
            for apply_quant in apply_quant_list:
                linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
                if isinstance(apply_quant, AOBaseConfig):
                    quantize_(linear, apply_quant)
                    ql = linear
                else:
                    # TODO(#1690): delete this once config migration is done
                    ql = apply_quant(linear)
                with tempfile.NamedTemporaryFile() as f:
                    torch.save(ql.state_dict(), f)
                    f.seek(0)
                    # `weights_only=True` is enabled for torch 2.5+
                    if TORCH_VERSION_AT_LEAST_2_5:
                        _ = torch.load(f, weights_only=True)
                    else:
                        _ = torch.load(f, weights_only=False)

    @unittest.skipIf(len(GPU_DEVICES) == 0, "Need GPU available")
    @common_utils.parametrize("apply_quant", get_quantization_functions(False, False))
    def test_to_device(self, apply_quant):
        for device in self.GPU_DEVICES:

            def _apply(module, config_or_subclass_inserter):
                if isinstance(config_or_subclass_inserter, AOBaseConfig):
                    quantize_(module, config_or_subclass_inserter)
                else:
                    # TODO(#1690): delete this once config migration is done
                    module = config_or_subclass_inserter(module)
                return module

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            ql = _apply(linear, apply_quant)
            ql.to(device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            ql = _apply(linear, apply_quant)
            ql.to(device=device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            ql = _apply(linear, apply_quant)
            ql.to(device)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_register_new_dispatch(self):
        from torchao.dtypes import AffineQuantizedTensor
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

    @skip_if_rocm("ROCm enablement in progress")
    @unittest.skipIf(len(GPU_DEVICES) == 0, "Need GPU available")
    def test_print_quantized_module(self):
        for device in self.GPU_DEVICES:
            apply_quant_list = get_quantization_functions(True, True, device, True)
            for apply_quant in apply_quant_list:
                linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
                if isinstance(apply_quant, AOBaseConfig):
                    quantize_(linear, apply_quant)
                    ql = linear
                else:
                    # TODO(#1690): delete this once config migration is done
                    ql = apply_quant(linear)
                assert "AffineQuantizedTensor" in str(ql)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize(
        "apply_quant", get_quantization_functions(False, True, "cuda", False)
    )
    def test_test_copy__apply(self, apply_quant):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        linear2 = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")

        if isinstance(apply_quant, AOBaseConfig):
            quantize_(linear, apply_quant)
            ql = linear
            quantize_(linear2, apply_quant)
            ql2 = linear2
        else:
            ql = apply_quant(linear)
            ql2 = apply_quant(linear2)

        example_input = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")
        output = ql(example_input)
        ql2.weight.copy_(ql.weight)
        ql2.bias = ql.bias
        output2 = ql2(example_input)
        self.assertEqual(output, output2)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize(
        "apply_quant", get_quantization_functions(False, True, "cuda", False)
    )
    def test_copy__mismatch_metadata(self, apply_quant):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        linear2 = torch.nn.Linear(128, 512, dtype=torch.bfloat16, device="cuda")

        if isinstance(apply_quant, AOBaseConfig):
            quantize_(linear, apply_quant)
            ql = linear
            quantize_(linear2, apply_quant)
            ql2 = linear2
        else:
            ql = apply_quant(linear)
            ql2 = apply_quant(linear2)

        # copy should fail due to shape mismatch
        with self.assertRaisesRegex(
            ValueError, "Not supported args for copy_ due to metadata mistach:"
        ):
            ql2.weight.copy_(ql.weight)

    def test_to_affine_quantized_intx_static(self):
        to_affine_quantized_intx_static(
            torch.randn(2, 3),
            scale=torch.randn(1),
            zero_point=torch.zeros(1),
            block_size=(2, 3),
            target_dtype=torch.int8,
            _layout=PlainLayout(),
        )


class TestAffineQuantizedBasic(TestCase):
    COMMON_DEVICES = (
        ["cpu"]
        + (["cuda"] if torch.cuda.is_available() else [])
        + (["xpu"] if torch.xpu.is_available() else [])
    )
    COMMON_DTYPES = [torch.bfloat16]

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    @skip_if_rocm("ROCm enablement in progress")
    def test_flatten_unflatten(self, device, dtype):
        if device == "cuda" and dtype == torch.bfloat16 and is_fbcode():
            raise unittest.SkipTest("TODO: Failing for cuda + bfloat16 in fbcode")
        apply_quant_list = get_quantization_functions(False, True, device)
        for apply_quant in apply_quant_list:
            linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
            if isinstance(apply_quant, AOBaseConfig):
                quantize_(linear, apply_quant)
                ql = linear
            else:
                # TODO(#1690): delete this once config migration is done
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

    @common_utils.parametrize("device", COMMON_DEVICES)
    @common_utils.parametrize("dtype", COMMON_DTYPES)
    def test_alias(self, device, dtype):
        dummy = nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(dummy, Int8DynamicActivationInt8WeightConfig())
        _ = dummy.weight[...]

    @common_utils.parametrize("device", ["cuda"])
    @common_utils.parametrize("dtype", [torch.bfloat16])
    @skip_if_no_cuda()
    def test_slice_int4wo(self, device, dtype):
        # in_feature not divisible by 1024
        # out_feature not divisible by 8
        # to test slice + padding for int4 weight only quantization
        dummy = nn.Linear(256, 321, dtype=dtype, device=device)
        quantize_(dummy, Int4WeightOnlyConfig())
        # make sure these run without error
        _ = dummy.weight.narrow(0, 0, 64)
        _ = dummy.weight.narrow(1, 0, 128)

    @common_utils.parametrize("device", ["cuda"])
    @common_utils.parametrize("dtype", [torch.float16, torch.bfloat16])
    @skip_if_no_cuda()
    @skip_if_no_gemlite()
    def test_slice_gemlite(self, device, dtype):
        # in_feature not divisible by 1024
        # out_feature not divisible by 8
        # to test slice + padding for int4 weight only quantization
        dummy = nn.Linear(256, 512, dtype=dtype, device=device)
        quantize_(dummy, GemliteUIntXWeightOnlyConfig())
        # make sure these run without error
        _ = dummy.weight.narrow(0, 0, 64)
        _ = dummy.weight.narrow(1, 0, 128)

    @common_utils.parametrize("device", ["cuda"])
    @common_utils.parametrize("dtype", [torch.bfloat16])
    def test_matmul(self, device, dtype):
        x = torch.randn(53, 2048)
        w = torch.randn(53, 2048)
        w = to_affine_quantized_intx(
            w,
            mapping_type=MappingType.SYMMETRIC,
            block_size=(1, 32),
            target_dtype=torch.int8,
            quant_min=-8,
            quant_max=7,
            eps=torch.finfo(torch.float32).eps,
        )
        # make sure it runs
        torch.matmul(x, w.t())


common_utils.instantiate_parametrized_tests(TestAffineQuantized)
common_utils.instantiate_parametrized_tests(TestAffineQuantizedBasic)


if __name__ == "__main__":
    run_tests()
