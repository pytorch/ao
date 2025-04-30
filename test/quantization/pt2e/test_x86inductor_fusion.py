# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: quantization"]
import contextlib
import copy
import itertools
import unittest

import torch
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase, run_tests
from torch._inductor.utils import run_and_get_code
from torch.export import export_for_training
from torch.testing._internal.common_quantization import (
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
    skipIfNoONEDNNBF16,
)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_LINUX,
    IS_X86,
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    HAS_CPU,
    _check_has_dynamic_shape,
)

import torchao
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_6,
    TORCH_VERSION_AT_LEAST_2_8,
)

if TORCH_VERSION_AT_LEAST_2_6:
    from torch.testing._internal.common_utils import TEST_ACL
else:
    TEST_ACL = False

# The dict value is match_nodes(computation_op+unary_op)
unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    # Use floats for min/max, otherwise they can get converted to symints
    torch.nn.Hardtanh(min_val=-0.5, max_val=4.0, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 3,
    torch.nn.Hardsigmoid(): 5,
}

non_decomposed_unary_list = [
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
]

# The dict value is (match_count, match_nodes, inplace)
binary_list = {
    lambda x, y: torch.add(x, y): (1, 2, False),  # call_function
    lambda x, y: torch.add(y, x): (1, 2, False),  # call_function
    lambda x, y: x.add(y): (1, 2, False),  # call_method
    lambda x, y: x.add_(y): (1, 2, True),  # call_method
    lambda x, y: torch.sub(x, y): (1, 2, False),  # call_function
    lambda x, y: x.sub(y): (1, 2, False),  # call_method
    lambda x, y: x.sub_(y): (1, 2, True),  # call_method
}

quantization_add_fn_list = [
    lambda x, y: torch.add(x, y),
    lambda x, y: x.add(y),
]

quantization_inplace_add_fn_list = [
    lambda x, y: x.add_(y),
]


def get_default_quantizer(is_qat, is_dynamic):
    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        xiq.get_default_x86_inductor_quantization_config(
            is_qat=is_qat, is_dynamic=is_dynamic
        )
    )
    return quantizer


def _generate_qdq_quantized_model(
    mod, inputs, is_qat=False, is_dynamic=False, quantizer=None
):
    maybe_no_grad = contextlib.nullcontext() if is_qat else torch.no_grad()
    with maybe_no_grad:
        export_model = export_for_training(mod, inputs, strict=True).module()
        quantizer = (
            quantizer if quantizer else get_default_quantizer(is_qat, is_dynamic)
        )
        prepare_model = (
            prepare_qat_pt2e(export_model, quantizer)
            if is_qat
            else prepare_pt2e(export_model, quantizer)
        )
        prepare_model(*inputs)
        torchao.quantization.pt2e.move_exported_model_to_eval(prepare_model)
        convert_model = convert_pt2e(prepare_model)
        return convert_model


def cal_conv_generated_kernel_number(mod, input, dtype, dim=4, device="cpu"):
    # this function is to decide how many kernels are generated
    # while testing conv2d/3d/deconv2d
    # the assumption is:
    #   (1) There will be a to_dtype kernel for input for lp
    #   (2) inductor always use channel_last format, there will
    #       be a to_channel_last format for input
    #   (3) to_dtype and to_channel_last for input can be fused
    #   (4) inductor always get channel last format from mkldnn_conv_pointwise(binary),
    #       and force the output to have same stride with eager.
    #       So there will be a to_contiguous for output if eager output is contiguouse
    mod = copy.deepcopy(mod)
    mod = mod.to(device=device)
    input = input.clone()
    input = input.to(device)

    if dtype == torch.float32:
        maybe_autocast = contextlib.nullcontext()
    else:
        maybe_autocast = torch.amp.autocast(device_type=device, dtype=dtype)
    with torch.no_grad(), maybe_autocast:
        output = mod(input)
    input_kernel, output_kernel = 0, 0
    if (
        input.is_contiguous(memory_format=torch.contiguous_format)
        or dtype != torch.float32
        or (TEST_ACL and dim == 4)
    ):
        input_kernel = 1
    if output.is_contiguous(memory_format=torch.contiguous_format) or (
        TEST_ACL and dtype == torch.bfloat16
    ):
        output_kernel = 1
    return input_kernel + output_kernel


@config.patch({"freezing": True})
class TestPatternMatcherBase(TestCase):
    def _check_unary_is_decomposed(self, unary_fn):
        return not any(
            isinstance(unary_fn, fn)
            for fn in [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]
        )

    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_check_fn,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        check_quantization=False,
        is_qat=False,
        dtype=None,
        is_dynamic=False,
        quantizer=None,
        compile_options={},  # noqa: B006
    ):
        if not hasattr(self, "device"):
            has_xpu = any(
                isinstance(input, torch.Tensor) and input.device.type == "xpu"
                for input in inputs
            )
            device = "xpu" if has_xpu else "cpu"
        else:
            device = self.device

        mod = mod.to(device=device)
        counters.clear()
        torch._dynamo.reset()
        if check_autocast == torch.bfloat16 and (
            torch.ops.mkldnn._is_mkldnn_bf16_supported() or device == "xpu"
        ):
            maybe_autocast = torch.amp.autocast(
                device_type=device, dtype=torch.bfloat16
            )
            atol, rtol = 1e-2, 1e-2
        elif check_autocast == torch.float16 and (
            torch.ops.mkldnn._is_mkldnn_fp16_supported() or device == "xpu"
        ):
            maybe_autocast = torch.amp.autocast(device_type=device, dtype=torch.float16)
            atol, rtol = 1e-2, 1e-2
        else:
            assert check_autocast == torch.float32
            maybe_autocast = contextlib.nullcontext()
        if check_quantization:
            convert_model = _generate_qdq_quantized_model(
                mod, inputs, is_qat, is_dynamic, quantizer
            )
            with torch.no_grad(), maybe_autocast:
                _ = torch.compile(convert_model)(*inputs)
                matcher_check_fn()
        else:
            with torch.no_grad(), maybe_autocast:
                clone_inputs = self._clone_inputs(inputs)
                expected = mod(*inputs)
                actual = torch.compile(mod, **compile_options)(*clone_inputs)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                matcher_check_fn()

    def _test_code_common(
        self,
        mod,
        inputs,
        include_ops,
        exclude_ops,
        atol=1e-5,
        rtol=1.3e-6,
        check_quantization=False,
        check_dynamic=None,
        num_include_ops=None,
        quantizer=None,
    ):
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            if check_quantization:
                mod = _generate_qdq_quantized_model(mod, inputs, quantizer=quantizer)
            expected = mod(*inputs)
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True, dynamic=check_dynamic),
                *clone_inputs,
            )
            for op in include_ops:
                self.assertIn(op, source_code)
            if num_include_ops is not None:
                assert len(include_ops) == len(num_include_ops)
                for i in range(len(include_ops)):
                    self.assertEqual(
                        source_code.count(include_ops[i]), num_include_ops[i]
                    )
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            if not check_quantization:
                # Skip due to reduce range setting for Quantization on preCI system.
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Requires torch 2.8+")
class TestPatternMatcher(TestPatternMatcherBase):
    def _qconv2d_test_helper(self, device="cpu", int8_mixed_bf16=False):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(
                    128, 128, kernel_size=3, stride=1, groups=4
                )

            def forward(self, x):
                return self.conv3(self.conv2(self.conv(x)))

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in QConv2D weight prepack * 1
            #    int8_mixed_fp32: [dequant_node, dequantize_per_channel, clone, convolution]
            #    int8_mixed_bf16: [dequant_node, optional(convert_element_type_4),
            #     dequantize_per_channel, optional(convert_element_type_3), clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 3
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"],
                18 if int8_mixed_bf16 else 12,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 3
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_cpu(self):
        r"""
        This testcase will quantize a single Conv2d module.
        """
        self._qconv2d_test_helper("cpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.
        """
        self._qconv2d_test_helper(int8_mixed_bf16=True)

    def _qconv2d_unary_test_helper(
        self,
        device="cpu",
        int8_mixed_bf16=False,
        unary_op=torch.nn.ReLU(),
        qconv_unary_matcher_nodes=None,
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.conv2 = torch.nn.Conv2d(
                    128, 128, kernel_size=3, stride=1, bias=False
                )
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.conv(x))
                return self.unary_fn2(self.conv2(tmp))

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 2
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )
            if qconv_unary_matcher_nodes:
                self.assertEqual(
                    counters["inductor"]["qconv_unary_matcher_nodes"],
                    0 if TEST_ACL else qconv_unary_matcher_nodes,
                )

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            matcher_check_fn=matcher_check_fn,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_relu_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu")

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_relu_int8_mixed_bf16_xpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qconv2d_unary_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_relu6_cpu(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_hardtanh_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.Hardtanh())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_hardtanh_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, clamp_min, clamp_max, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            unary_op=torch.nn.Hardtanh(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=11,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_hardswish_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.Hardswish())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_hardswish_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min,
             clamp_max, mul, div, convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, add, clamp_min, clamp_max, mul, div, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            unary_op=torch.nn.Hardswish(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=17,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_silu_cpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        """
        self._qconv2d_unary_test_helper(device="cpu", unary_op=torch.nn.SiLU())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_silu_int8_mixed_bf16_cpu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern.
        Match.nodes:
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul,
             convert_element_type, quantize_per_tensor]
            [qconv2d_pointwise_default, convert_element_type, sigmoid, mul, convert_element_type]
        """
        self._qconv2d_unary_test_helper(
            unary_op=torch.nn.SiLU(),
            int8_mixed_bf16=True,
            qconv_unary_matcher_nodes=11,
        )

    def _qconv2d_add_test_helper(
        self, device="cpu", use_relu=False, int8_mixed_bf16=False
    ):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, bias=False)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, bias=False)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        for add_fn in quantization_add_fn_list + quantization_inplace_add_fn_list:
            mod = M(add_fn, use_relu).eval().to(device=device)
            v = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
                .add(1)
                .to(device=device)
            )

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 4
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 4
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 2
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            )

    def _qconv2d_add_test_helper2(
        self, device="cpu", use_relu=False, int8_mixed_bf16=False
    ):
        r"""
        This testcase will quantize two Conv2d->Add patterns as:

        Conv(X)   extra input
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y

        , and

        extra input   Conv(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                swap_inputs,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1, bias=False)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu
                self.swap_inputs = swap_inputs

            def forward(self, x, x2, x3):
                x1 = self.conv1(x)
                if self.swap_inputs:
                    tmp = self.add_fn(x2, x1)
                else:
                    tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv2(tmp)
                if self.swap_inputs:
                    res = self.add_fn2(x3, tmp1)
                else:
                    res = self.add_fn2(tmp1, x3)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        for add_fn, swap_inputs in itertools.product(
            quantization_add_fn_list + quantization_inplace_add_fn_list, [False, True]
        ):
            mod = M(add_fn, use_relu, swap_inputs).eval().to(device=device)
            x = torch.randn(
                (1, 3, 8, 8), dtype=torch.float32, requires_grad=False, device=device
            )
            x2 = torch.randn(
                (1, 6, 6, 6), dtype=torch.float32, requires_grad=False, device=device
            )
            x3 = torch.randn(
                (1, 6, 4, 4), dtype=torch.float32, requires_grad=False, device=device
            )

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 2
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                (x, x2, x3),
                matcher_check_fn,
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_cpu(self):
        self._qconv2d_add_test_helper()
        self._qconv2d_add_test_helper2()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_add_int8_mixed_bf16(self):
        self._qconv2d_add_test_helper(int8_mixed_bf16=True)
        self._qconv2d_add_test_helper2(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_relu_cpu(self):
        self._qconv2d_add_test_helper(use_relu=True)
        self._qconv2d_add_test_helper2(use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qconv2d_add_relu_int8_mixed_bf16(self):
        self._qconv2d_add_test_helper(use_relu=True, int8_mixed_bf16=True)
        self._qconv2d_add_test_helper2(use_relu=True, int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_broadcast_shapes_cpu(self):
        r"""
        This testcase will quantize Conv2d->add pattern using broadcast shape inputs.
        Conv2d->Add fusion will fail for the broadcast shape inputs case.
        """

        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1)

            def forward(self, x1, x2):
                return torch.add(self.conv(x1), x2)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 32, 9, 9))
            x2 = torch.randn((2, 32, 1, 1))

            def matcher_check_fn():
                # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 1
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 1
                )
                # 2. Qconv2d Binary Unary fusion in post-grad fusion pass * 0
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (x1, x2),
                matcher_check_fn,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_with_concat_cpu(self):
        channel_1 = 32
        channel_2 = 16
        channel_3 = 8
        channel_4 = int(channel_2 * 2 + channel_3)

        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    channel_1, channel_2, 1, stride=1, dilation=1, padding=0
                )
                self.conv2 = torch.nn.Conv2d(
                    channel_1, channel_2, 1, stride=1, dilation=1, padding=0
                )
                self.conv3 = torch.nn.Conv2d(
                    channel_2, channel_3, 3, stride=1, dilation=1, padding=1
                )

                self.conv = torch.nn.Conv2d(
                    channel_4, channel_2, 1, stride=1, dilation=1, padding=0
                )

            def forward(self, x: torch.Tensor):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                x3 = self.conv3(x2)
                res = torch.cat([x1, x2, x3], dim=1)
                res = self.conv(res)
                return res

        mod = Model().eval()
        v = torch.randn(
            (8, channel_1, 40, 40), dtype=torch.float32, requires_grad=False
        )

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 4
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 3,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 4
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_2(self):
        r"""
        This testcase prevents this pattern be matched as a conv_binary fusion by mistake.
                Conv(X)  3
                    \   /
                     Add
        We see this pattern in Mobilenet v3 large which add is decomposed from torch.nn.Hardswish or torch.nn.Hardsigmoid.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                post_op,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.post_op = post_op

            def forward(self, x):
                return self.post_op(self.conv(x))

        for post_op in [
            torch.nn.Hardswish(inplace=True),
            torch.nn.Hardsigmoid(inplace=True),
        ]:
            mod = M(post_op).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            def matcher_check_fn():
                # Shouldn't hit conv binary fusion
                self.assertEqual(
                    counters["inductor"]["qconv2d_binary_matcher_count"], 0
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv2d_add_3(self):
        r"""
        This testcase will test below model:
             x
           /   \
        conv1  maxpool
          \    /   \
           add    conv2
            \     /
              cat
        Based on default recipe of x86InductorQuantizer, we will see this pattern after convert:
        qconv1    maxpool
         \           |
          \         q1
           \       /   \
            \     dq1  qconv2
             \   /
              add
               |
               q2
        Since q1 has 2 users and qconv2 is not ancestor node of qconv1, we shouldn't fuse:
                int8
                 /
        qconv1 dq1
           \   /
            add
             |
             q2
             |
            int8
        Instead we can match and fuse this pattern into qconv_binary:
        qconv1  fp32
            \   /
             add
              |
             fp32
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
                self.maxpool = torch.nn.MaxPool2d(
                    kernel_size=3, stride=1, padding=0, dilation=1
                )

            def forward(self, x):
                tmp1 = self.conv1(x)
                tmp2 = self.maxpool(x)
                add = torch.add(tmp1, tmp2)
                tmp3 = self.conv2(tmp2)
                return torch.cat((add, tmp3), dim=1)

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            # The matched qconv binary pattern should have 2 nodes [qconv, add]
            # instead of 11 which has dequant in binary input and output quant
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d(self):
        r"""
        This testcase will quantize a single Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.bn = torch.nn.BatchNorm2d(128)

            def forward(self, x):
                return self.bn(self.conv(x))

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 1
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 4
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, quantize_per_tensor]
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_nodes"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 1
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    def _qat_qconv2d_unary_cpu_test_helper(
        self,
        unary_op=torch.nn.ReLU(),
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.bn = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
                self.unary_fn2 = copy.deepcopy(unary_op)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                tmp = self.unary_fn(self.bn(self.conv(x)))
                return self.unary_fn2(self.bn2(self.conv2(tmp)))

        mod = M()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 1
            #    [convert_element_type_1, sub, mul_1, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default, relu, div_1, round_2, add_1, clamp_min_1, clamp_max_1, convert_element_type_2]
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_relu(self):
        r"""
        This testcase will quantize Conv2d->ReLU pattern with qat flow.
        """

        self._qat_qconv2d_unary_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_relu6(self):
        r"""
        This testcase will quantize Conv2d->ReLU6 pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.ReLU6())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_hardtanh(self):
        r"""
        This testcase will quantize Conv2d->Hardtanh pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardtanh())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_silu(self):
        r"""
        This testcase will quantize Conv2d->SiLU pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.SiLU())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_qconv2d_hardswish(self):
        r"""
        This testcase will quantize Conv2d->Hardswish pattern with qat flow.
        """
        self._qat_qconv2d_unary_cpu_test_helper(unary_op=torch.nn.Hardswish())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add(self):
        r"""
        This testcase will quantize a Conv2d->Add pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)

            def forward(self, x):
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return x1 + x2

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 8
            )
            # 2. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, dequantize_per_tensor, add_3, quantize_per_tensor]
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 4,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add_relu(self):
        r"""
        This testcase will quantize a Conv2d->Add->ReLU pattern as:
                 X
               /   \
        Conv1(X)   Conv2(X)
               \   /
                Add
                 |
                ReLU
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return self.relu(x1 + x2)

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            # 1. Dequant-conv pattern matched in quantization weight prepack * 2
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 8
            )
            # 2. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, dequantize_per_tensor, add_3, relu, quantize_per_tensor]
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 5,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    def _test_qconv2d_dequant_promotion_helper(self, device="cpu"):
        r"""
        This testcase tests if dequant node before conv2d is promoted correctly:
                 X
                 |
              Conv1(X)
               /   \
        Conv2(X)   Conv3(X)
               \   /
                Add
                 |
                 Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)

            def forward(self, x):
                temp = self.conv1(x)
                temp = self.conv2(temp) + self.conv3(temp)
                return temp

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            #    [dequantize_per_tensor]
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_nodes"], 1)
            # 2. Dequant-conv pattern matched in quantization weight prepack * 3
            #    [dequantize_per_tensor, dequantize_per_channel, clone, convolution]
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 3
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_nodes"], 12
            )
            # 3. Qconv2d Binary fusion in post-grad fusion pass * 1
            #    [qconv2d_pointwise_default_1, add_3]
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_count"],
                0 if TEST_ACL else 1,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_matcher_nodes"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv2d_binary_lower_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_dequant_promotion_cpu(self):
        self._test_qconv2d_dequant_promotion_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qconv1d_relu_cpu(self):
        r"""
        This testcase will quantize Conv1d->ReLU pattern.
        """
        device = "cpu"
        unary_op = torch.nn.ReLU()

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = copy.deepcopy(unary_op)
                self.conv2 = torch.nn.Conv1d(
                    128, 128, kernel_size=3, stride=1, bias=False
                )
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.conv(x))
                return self.unary_fn2(self.conv2(tmp))

        mod = M().eval().to(device=device)
        v = (
            torch.randn((1, 3, 8), dtype=torch.float32, requires_grad=False)
            .add(1)
            .to(device=device)
        )

        def matcher_check_fn():
            # 1. Dequant-Conv2D pattern matched in quantization weight prepack * 2
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            # 2. QConv2D Unary fusion in post-grad fusion pass * 2
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )

        self._test_common(
            mod,
            (v,),
            check_quantization=True,
            matcher_check_fn=matcher_check_fn,
        )

    def _qlinear_test_helper(
        self,
        inputs,
        device="cpu",
        int8_mixed_bf16=False,
        do_permute=False,
        matcher_check_fn=None,
        bias=True,
        is_dynamic=False,
        is_qat=False,
    ):
        class M(torch.nn.Module):
            def __init__(self, use_bias, do_permute=False):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3, use_bias)
                self.linear2 = torch.nn.Linear(3, 4, use_bias)
                self.do_permute = do_permute

            def forward(self, x):
                if self.do_permute:
                    x = torch.reshape(torch.permute(x, (0, 2, 3, 1)), (2, 12, 4))
                return self.linear2(self.linear(x))

        mod = M(bias, do_permute=do_permute).eval().to(device=device)
        assert isinstance(inputs, tuple)

        def __convert_tensor_to_device(input, device):
            return input.to(device=device) if isinstance(input, torch.Tensor) else input

        inputs = tuple(__convert_tensor_to_device(input, device) for input in inputs)

        def _default_matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
            )

        self._test_common(
            mod,
            inputs,
            matcher_check_fn=(
                matcher_check_fn
                if matcher_check_fn is not None
                else _default_matcher_check_fn
            ),
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            check_quantization=True,
            is_qat=is_qat,
            is_dynamic=is_dynamic,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_cpu(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_test_helper((torch.randn((2, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_dynamic_qlinear_cpu(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),), bias=bias, is_dynamic=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_dynamic_qlinear_qat_cpu(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),), bias=bias, is_dynamic=True, is_qat=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_dynamic_qlinear_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)),), bias=bias, is_dynamic=True
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a single Linear Moduel with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Moduel.
        """
        for bias in [True, False]:
            self._qlinear_test_helper((torch.randn((2, 3, 4)),), bias=bias)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a single Linear Moduel with int8_mixed_bf16 quantization.
        """
        for bias in [True, False]:
            self._qlinear_test_helper(
                (torch.randn((2, 3, 4)),), int8_mixed_bf16=True, bias=bias
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    13 if bias else 12,
                )

            self._qlinear_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_int8_mixed_bf16_input_dim_exceeds_2_and_not_contiguous(self):
        r"""
        This testcase will quantize a single Linear Module for int8_bf16.
        * Input dim exceeds 2
        * Input not contiguous
        """
        for bias in [True, False]:

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    17 if bias else 16,
                )

            self._qlinear_test_helper(
                (torch.randn((2, 4, 3, 4)),),
                int8_mixed_bf16=True,
                do_permute=True,
                matcher_check_fn=matcher_check_fn,
                bias=bias,
            )

    def _qlinear_unary_test_helper(
        self, inputs, unary_op=torch.nn.ReLU(), device="cpu", int8_mixed_bf16=False
    ):
        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn = copy.deepcopy(unary_op)
                self.linear2 = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn2 = copy.deepcopy(unary_op)

            def forward(self, x):
                tmp = self.unary_fn(self.linear(x))
                return self.unary_fn2(self.linear2(tmp))

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval().to(device=device)

            def matcher_check_fn():
                # 1. dequant-linear pattern matched in quantization weight prepack
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 2
                )
                # 2. QLinear Unary fusion in post-grad fusion pass
                self.assertEqual(
                    counters["inductor"]["qlinear_unary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_unary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                inputs,
                matcher_check_fn,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_relu_cpu(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_relu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 4)),), int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_relu_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 3, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_relu_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.
        """
        self._qlinear_unary_test_helper((torch.randn((2, 3, 4)),), int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_gelu_cpu(self):
        r"""
        This testcase will quantize a Linear->GELU pattern.
        """
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            self._qlinear_unary_test_helper((torch.randn((2, 4)),), gelu)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_gelu_int8_mixed_bf16(self):
        r"""
        This testcase will quantize a Linear->GELU pattern with int8_mixed_bf16 quantization.
        """
        for gelu in [torch.nn.GELU("none"), torch.nn.GELU("tanh")]:
            self._qlinear_unary_test_helper(
                (torch.randn((2, 4)),), gelu, int8_mixed_bf16=True
            )

    def _qlinear_add_test_helper(
        self,
        device="cpu",
        use_relu=False,
        int8_mixed_bf16=False,
        is_qat=True,
        is_dynamic=True,
    ):
        r"""
        This testcase will quantize two consecutive Linear->Add(->relu) patterns as:
                 X
               /   \
        linear(X)   linear(X)
               \   /
                Add
                 |
           Optional(relu)
               /   \
        linear(X)   linear(X)
               \   /
                Add
                 |
           Optional(relu)
                 |
                 Y
        """

        def fake_quant(x):
            # to produce a float32 result as extra input
            qlib = torch.ops.quantized_decomposed
            if device == "cpu":
                qmin, qmax, dtype = 0, 255, torch.uint8
            else:
                qmin, qmax, dtype = -128, 127, torch.int8
            x = qlib.quantize_per_tensor.default(x, 0.0166785, 42, qmin, qmax, dtype)
            x = qlib.dequantize_per_tensor.default(x, 0.0166785, 42, qmin, qmax, dtype)
            return x

        class M(torch.nn.Module):
            def __init__(
                self,
                add_fn,
                use_relu,
                fake_quant_before_extra_input,
            ):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.linear3 = torch.nn.Linear(4, 4)
                self.linear4 = torch.nn.Linear(4, 4)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu
                self.fake_quant_before_extra_input = fake_quant_before_extra_input

            def forward(self, x):
                x1 = self.linear1(x)
                x2 = self.linear2(x)
                if self.fake_quant_before_extra_input:
                    x2 = fake_quant(x2)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.linear3(tmp)
                tmp2 = self.linear4(tmp)
                if self.fake_quant_before_extra_input:
                    tmp2 = fake_quant(tmp2)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        add_fn_list = [
            lambda x, y: x + y,
            lambda x, y: y + x,
            lambda x, y: x.add_(y),
            lambda x, y: y.add_(x),
        ]
        fake_quant_x2_list = [False, True] if int8_mixed_bf16 else [False]
        shape_list = [(4, 4), (4, 4, 4)]
        cases = itertools.product(add_fn_list, fake_quant_x2_list, shape_list)
        for add_fn, fq_x2, shape in cases:
            mod = M(add_fn, use_relu, fq_x2).eval().to(device=device)
            v = torch.randn(
                shape, dtype=torch.float32, requires_grad=False, device=device
            ).add(1)

            def matcher_check_fn():
                # 1. Dequant-linear pattern matched in quantization weight prepack * 4
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 4
                )
                # pattern = [dequant_per_tensor, (convert_dtype), dequant_per_channel, (convert_dtype), permute, addmm]
                nodes_per_match = 6 if int8_mixed_bf16 else 4
                if len(shape) == 3:
                    # pattern = [dequant_per_tensor, (convert_dtype), (view), \
                    #   dequant_per_channel, (convert_dtype), (view), permute, addmm]
                    nodes_per_match += 2
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    4 * nodes_per_match,
                )
                # 2. Qlinear Binary Unary fusion in post-grad fusion pass * 2
                self.assertEqual(
                    counters["inductor"]["qlinear_binary_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                # Two linear-binary patterns are matched
                # matched patter1 = [qlinear, add, (convert dtype), (relu), quantize_per_tensor]
                # matched patter2 = [qlinear, add, (convert dtype), (relu)]
                # If add_fn is x.add_(y), x is bf16 and y is fp32, there is a to_bf16 node after binary
                to_bf16_after_binary = 2 * (add_fn == add_fn_list[2] and fq_x2)
                expected_matcher_nodes = (
                    (4 if is_dynamic else 5) + 2 * use_relu + to_bf16_after_binary
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_binary_matcher_nodes"],
                    0 if TEST_ACL else expected_matcher_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_binary_lower_count"],
                    0 if TEST_ACL else 2,
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
                check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
                is_qat=is_qat,
                is_dynamic=is_dynamic,
            )

            if TEST_ACL:
                continue

            if torch._inductor.config.cpp_wrapper:
                # For CPP wrapper
                self._test_code_common(
                    mod,
                    (v,),
                    [
                        "aoti_torch_cpu__qlinear_pointwise_tensor",
                        "aoti_torch_cpu__qlinear_pointwise_binary_tensor",
                    ],
                    [],
                    check_quantization=True,
                    num_include_ops=[2, 2],
                )
            else:
                # For python wrapper
                self._test_code_common(
                    mod,
                    (v,),
                    [
                        "torch.ops.onednn.qlinear_pointwise.tensor",
                        "torch.ops.onednn.qlinear_pointwise.binary",
                    ],
                    [],
                    check_quantization=True,
                    num_include_ops=[2, 2],
                )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @parametrize("use_relu", [True, False])
    @parametrize("is_qat", [True, False])
    @parametrize("is_dynamic", [True, False])
    def test_qlinear_add_cpu(self, use_relu, is_qat, is_dynamic):
        self._qlinear_add_test_helper(
            use_relu=use_relu, is_qat=is_qat, is_dynamic=is_dynamic
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @parametrize("use_relu", [True, False])
    @parametrize("is_qat", [True, False])
    @parametrize("is_dynamic", [True, False])
    def test_qlinear_add_int8_mixed_bf16(self, use_relu, is_qat, is_dynamic):
        self._qlinear_add_test_helper(
            int8_mixed_bf16=True,
            use_relu=use_relu,
            is_qat=is_qat,
            is_dynamic=is_dynamic,
        )

    def _qlinear_dequant_promotion_test_helper(
        self,
        inputs,
        device="cpu",
        int8_mixed_bf16=False,
        is_dynamic=False,
        matcher_check_fn=None,
    ):
        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.linear3 = torch.nn.Linear(4, 4)

            def forward(self, x):
                temp = self.linear1(x)
                temp = self.linear2(temp) + self.linear3(temp)
                return temp

        mod = M().eval().to(device=device)

        def default_matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            # 2. dequant-linear pattern matched in quantization weight prepack * 3
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 3
            )
            # 3. QLinear Unary fusion in post-grad fusion pass * 1
            self.assertEqual(
                counters["inductor"]["qlinear_unary_matcher_count"],
                0 if TEST_ACL else 1,
            )

        self._test_common(
            mod,
            inputs,
            matcher_check_fn=(
                matcher_check_fn
                if matcher_check_fn is not None
                else default_matcher_check_fn
            ),
            check_autocast=torch.bfloat16 if int8_mixed_bf16 else torch.float,
            check_quantization=True,
            is_dynamic=is_dynamic,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_cpu(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_test_helper((torch.randn((2, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_int8_mixed_bf16(self):
        r"""
        Test with int8_mixed_bf16 quantization.
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_test_helper(
            (torch.randn((2, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_cpu_input_dim_exceeds_2(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_test_helper((torch.randn((2, 3, 4)),))

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_int8_mixed_bf16_input_dim_exceeds_2(self):
        r"""
        Test with int8_mixed_bf16 quantization.
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """
        self._qlinear_dequant_promotion_test_helper(
            (torch.randn((2, 3, 4)),), int8_mixed_bf16=True
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_dequant_promotion_dynamic_cpu(self):
        r"""
        This testcase test if dequant node before linear is promoted correctly:
                  X
                  |
               Linear1(X)
                /   \
        Linear2(X)   Linear3(X)
                \   /
                 Add
                  |
                  Y
        """

        def matcher_check_fn():
            # 1. Dequant pattern matcher for dequant promotion * 1
            self.assertEqual(counters["inductor"]["dequant_promotion_matcher_count"], 1)
            # 2. dequant-linear pattern matched in quantization weight prepack * 3
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 3
            )

        self._qlinear_dequant_promotion_test_helper(
            (torch.randn((2, 4)),),
            matcher_check_fn=matcher_check_fn,
            is_dynamic=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qlinear_mul_cpu(self):
        r"""
        This testcase will quantize a Linear->Mul pattern.
        """

        class M(torch.nn.Module):
            def __init__(self, use_bias):
                super().__init__()
                self.linear = torch.nn.Linear(4, 5, use_bias)

            def forward(self, x1, x2):
                return torch.mul(self.linear(x1), x2)

        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 4))
            x2 = torch.randn((2, 5))

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 1
                )

            self._test_common(
                mod,
                (x1, x2),
                matcher_check_fn,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    def test_qmaxpool2d(self):
        r"""
        This testcase will quantize Conv2d->ReLU->MaxPool2d pattern.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3, **kwargs)

            def forward(self, x):
                return self.maxpool(self.relu(self.conv(x)))

        kwargs_list = [
            {"stride": 2},
            {"stride": 2, "padding": 1},
            {"stride": 2, "padding": 1, "dilation": 1},
            {"stride": 2, "padding": 1, "dilation": 1, "ceil_mode": False},
        ]
        for kwargs in kwargs_list:
            mod = M(kwargs).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                1
            )

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qmaxpool2d_matcher_count"],
                    0 if TEST_ACL else 1,
                )
                self.assertEqual(
                    counters["inductor"]["qconv_weight_prepack_matcher_count"], 1
                )
                self.assertEqual(
                    counters["inductor"]["qconv_unary_matcher_count"],
                    0 if TEST_ACL else 1,
                )
                self.assertEqual(
                    counters["inductor"]["qconv_unary_lower_count"],
                    0 if TEST_ACL else 1,
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
            )

    @skipIfNoDynamoSupport
    def test_qflatten(self):
        r"""
        This testcase will quantize Conv2d->AdaptiveAvgPool2d->flatten->cat pattern.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.relu = torch.nn.ReLU()
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                return torch.cat(
                    [
                        torch.flatten(
                            self.adaptive_avg_pool2d(self.relu(self.conv(x))), 1
                        )
                    ]
                )

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qreshape_matcher_count"], 0 if TEST_ACL else 1
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    @skipIfNoDynamoSupport
    def test_qcat(self):
        r"""
        This testcase will quantize cat based pattern:
                X
             /     \
        Conv1(X)  Pow(x)
            \        \
             \     Conv2(X)
              \    /
               Cat
                |
                Y
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )
                self.conv2 = torch.nn.Conv2d(
                    3, 64, 7, bias=True, stride=2, padding=3, dilation=1
                )

            def forward(self, x):
                temp1 = self.conv(x)
                temp2 = self.conv2(torch.pow(x, 2))
                return torch.cat((temp1, temp2), 1)

        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qcat_matcher_count"], 0 if TEST_ACL else 1
            )
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["qconv_unary_lower_count"], 0 if TEST_ACL else 2
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
        )

    def _test_linear_dynamic_fp16_helper(self, use_relu: bool):
        class M(torch.nn.Module):
            def __init__(self, bias: bool, use_relu: bool):
                super().__init__()
                self.linear = torch.nn.Linear(256, 256, bias=bias)
                self.relu = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                if self.use_relu:
                    return self.relu(self.linear(x))
                return self.linear(x)

        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        quantizer.set_module_type_qconfig(
            torch.nn.Linear, xiq.get_x86_inductor_linear_dynamic_fp16_config()
        )
        bias_list = [True, False]
        input_ndim_list = [2, 3]
        x_contig_list = [True, False]
        cases = itertools.product(bias_list, input_ndim_list, x_contig_list)
        for bias, input_ndim, x_contig in cases:
            x_shape = (4, 256) if input_ndim == 2 else (4, 1, 256)
            x = torch.randn(x_shape)
            if not x_contig:
                x = x[0::2, ...]
            mod = M(bias, use_relu).eval()

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 1
                )
                # Matched nodes:
                # (1) w to fp16, (2) w to fp32, (3) permute w, (4) mm/addmm/bmm
                # If x.ndim == 3 and x is contiguous, two view nodes are added.
                # If x.ndim == 3 and x is not contiguous, two expand nodes and one add node are added.
                nodes_count = 4
                if input_ndim > 2:
                    if x_contig:
                        nodes_count += 2
                    else:
                        nodes_count += 3 if bias else 2
                if use_relu:
                    nodes_count += 1
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                    nodes_count,
                )

            self._test_common(
                mod,
                (x,),
                atol=1e-2,
                rtol=1e-2,
                matcher_check_fn=matcher_check_fn,
                check_quantization=True,
                quantizer=quantizer,
            )
            linear_op_str = (
                "torch.ops.onednn.linear_relu_dynamic_fp16.default"
                if use_relu
                else "torch.ops.onednn.linear_dynamic_fp16.default"
            )
            self._test_code_common(
                mod,
                (x,),
                [linear_op_str],
                ["torch.ops.aten.addmm.default", "torch.ops.aten.mm.default"],
                check_quantization=True,
                quantizer=quantizer,
            )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_linear_dynamic_fp16(self):
        self._test_linear_dynamic_fp16_helper(use_relu=False)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_linear_relu_dynamic_fp16(self):
        self._test_linear_dynamic_fp16_helper(use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    # TODO: investigate options of torch.compile in fbcode
    @unittest.skipIf(IS_FBCODE, "Failing in fbcode")
    @parametrize("has_bias", [True, False])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("per_channel_quant", [True, False])
    @parametrize("dynamic", [True, False])
    def test_smooth_quant_with_int_mm(
        self, has_bias, dtype, per_channel_quant, dynamic
    ):
        r"""
        This testcase check if we can match the SmoothQuant int8 linear pattern from Torchao.
        The pattern is:
            (no bias) reshape -> _int_mm -> convert_element_type -> (expand -> mul) -> mul -> reshape
        or
            (with bias) pattern_no_bias -> add -> reshape -> reshape
        """
        if dtype == torch.bfloat16 and not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            return
        M = 16
        in_feature = 32
        out_feature = 64
        q_min, q_max = -32, 31

        class Mod(torch.nn.Module):
            def __init__(
                self, dtype: torch.dtype, has_bias: bool, per_channel_quant: bool
            ):
                super().__init__()
                self.dtype = dtype
                self.has_bias = has_bias
                self.b = torch.randint(
                    q_min, q_max, [in_feature, out_feature], dtype=torch.int8
                )
                self.per_channel_quant = per_channel_quant
                a_scale_per_tensor = torch.rand([1], dtype=dtype) * 0.01 + 0.01
                a_scale_per_channel = torch.rand([M, 1], dtype=dtype) * 0.01 + 0.01
                self.a_scale = (
                    a_scale_per_channel
                    if self.per_channel_quant
                    else a_scale_per_tensor
                )
                self.b_scale = torch.rand([out_feature]) * 0.01 + 0.01
                self.b_scale = self.b_scale.to(dtype)
                self.bias = torch.rand([out_feature], dtype=dtype) if has_bias else None

            def forward(self, a):
                out_shape = a.shape[:-1] + (self.b.size(-1),)
                a_reshaped = a.reshape(-1, a.size(-1))
                c = torch._int_mm(a_reshaped, self.b)
                c = c.to(self.dtype)
                c_shape = c.shape
                a_scale = self.a_scale.expand(c.shape)
                c = c * a_scale
                c = c * self.b_scale
                if self.has_bias:
                    c = c.reshape([1, *list(c_shape)])
                    c = c + self.bias
                    c = c.reshape(c_shape)
                c = c.reshape(out_shape)
                return c

        mod = Mod(dtype, has_bias, per_channel_quant).eval()
        a = torch.randint(q_min, q_max, [1, M, in_feature], dtype=torch.int8)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 1
            )
            if dynamic:
                nodes_count = 10 if has_bias else 7
            else:
                nodes_count = 7 if has_bias else 6
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_nodes"],
                nodes_count,
            )

        self._test_common(
            mod,
            (a,),
            matcher_check_fn=matcher_check_fn,
            check_autocast=dtype,
            compile_options={"dynamic": dynamic},
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    # TODO: investigate options of torch.compile in fbcode
    @unittest.skipIf(IS_FBCODE, "Failing in fbcode")
    @parametrize("has_bias", [True, False])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    @parametrize("dynamic", [True, False])
    @parametrize("reshape_a", [True, False])
    @parametrize(
        "M",
        [
            1,
            32,
        ],
    )
    @parametrize("inplace_add", [True, False])
    @parametrize("expand_a_scale", [True, False])
    def test_da8w8_sym_act_sym_wgt_with_int_mm(
        self, has_bias, dtype, dynamic, reshape_a, M, inplace_add, expand_a_scale
    ):
        r"""
        This testcase check if we can match the int8_dynamic_activation_int8_weight int8 linear pattern from torchao,
        when activation is symmetrically quantized dynamically & weights are symmetrically quantized (statically)
        The pattern is:
            (no bias) _int_mm -> convert_element_type -> ([expand_a] -> mul) -> mul
        or
            (with bias) pattern_no_bias -> add
        Expansion of the scale of activation is optional.
        The pattern depiction doesn't mean that convert_element_type output is fed into expand_a as input,
        but simply that activation scale may be applied after an expand operation on it.
        """
        if dtype == torch.bfloat16 and not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            return
        in_feature = 32
        out_feature = 64
        q_min, q_max = -32, 31
        # we only test for qlinear_binary in this case
        test_for_pointwise_binary = (
            True
            if M == 1
            and inplace_add
            and not expand_a_scale
            and not dynamic
            and not has_bias
            else False
        )
        if test_for_pointwise_binary and not IS_X86:
            self.skipTest("Some UTs are only supported on x86_64 CPUs")

        class Mod(torch.nn.Module):
            def __init__(self, dtype: torch.dtype, has_bias: bool):
                super().__init__()
                self.dtype = dtype
                self.has_bias = has_bias
                self.b = torch.randint(
                    q_min, q_max, [in_feature, out_feature], dtype=torch.int8
                )
                self.a_scale = torch.rand([M, 1], dtype=dtype) * 0.01 + 0.01
                self.b_scale = torch.rand([out_feature]) * 0.01 + 0.01
                self.b_scale = self.b_scale.to(dtype)
                self.bias = torch.rand([out_feature], dtype=dtype) if has_bias else None
                self.additive = torch.rand([M, out_feature], dtype=dtype)

            def forward(self, a):
                if reshape_a:
                    a_reshaped = a.reshape(-1, a.size(-1))
                else:
                    a_reshaped = a
                c = torch._int_mm(a_reshaped, self.b)
                c = c.to(self.dtype)
                if expand_a_scale:
                    a_scale = self.a_scale.expand(c.shape)
                else:
                    a_scale = self.a_scale
                c = c * a_scale
                c = c * self.b_scale
                if self.has_bias:
                    c = c + self.bias
                elif inplace_add and test_for_pointwise_binary:
                    # When M is 1, dynamic shapes are enabled with torch.compile, has_bias is False,
                    # expand_a_scale is False and inplace_add is true,
                    # the output's outermost dim's stride can't be determined due to some Inductor bug.
                    c.add_(self.additive)
                return c

        mod = Mod(dtype, has_bias).eval()
        a = torch.randint(q_min, q_max, [M, in_feature], dtype=torch.int8)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qlinear_weight_prepack_matcher_count"], 1
            )

        self._test_common(
            mod,
            (a,),
            matcher_check_fn,
            check_autocast=dtype,
            compile_options={"dynamic": dynamic},
        )
        if test_for_pointwise_binary:
            self.assertEqual(counters["inductor"]["qlinear_binary_matcher_count"], 1)


@dynamo_config.patch(
    {
        "dynamic_shapes": True,
        "assume_static_by_default": False,
        "specialize_float": True,
    }
)
@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Requires torch 2.8+")
class TestDynamicPatternMatcher(TestPatternMatcherBase):
    def test_qconv2d_maxpool2d_linear_dynamic_cpu(self, include_ops=None):
        r"""
        This testcase will quantize a single Conv2d->Maxpool2d->Linear module
        with dynamic batch size input.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
                **kwargs,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, (2, 2), stride=(1, 1), padding=(1, 1)
                )
                self.relu = torch.nn.ReLU()
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                temp = self.relu(self.conv(x))
                temp = self.maxpool2d(temp)
                temp = self.avgpool(temp)
                temp = torch.flatten(temp, 1)
                return self.linear(temp)

        mod = M().eval()
        v = torch.randn((2, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)
        if include_ops is None:
            include_ops = [
                "torch.ops.onednn.qconv_pointwise",
                "torch.ops.quantized.max_pool2d",
                "torch.ops.onednn.qlinear_pointwise",
            ]
        exclude_ops = []
        self._test_code_common(
            mod,
            (v,),
            include_ops,
            exclude_ops,
            check_quantization=True,
            check_dynamic=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_qat_bn_conv2d(self):
        r"""
        This testcase will quantize a single BN Conv2d module with qat flow.
        """

        class M(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.bn2 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(self.bn1(x))
                return self.bn2(x)

        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            self.assertEqual(
                counters["inductor"]["qconv_weight_prepack_matcher_count"], 1
            )

        self._test_common(
            mod,
            (v,),
            matcher_check_fn,
            check_quantization=True,
            is_qat=True,
        )

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_q_attention_block(self):
        class SelfAttnLikeModule(torch.nn.Module):
            def __init__(
                self,
                input_dim,
                transpose_for_score=False,
                num_attention_heads=None,
                attention_head_size=None,
            ) -> None:
                super().__init__()
                self.input_dim = input_dim
                self.q_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
                self.k_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
                self.v_proj = torch.nn.Linear(input_dim, input_dim, bias=False)
                self.softmax = torch.nn.Softmax(dim=-1)
                self.transpose_for_score = transpose_for_score
                if self.transpose_for_score:
                    assert num_attention_heads is not None
                    assert attention_head_size is not None
                    self.num_attention_heads = num_attention_heads
                    self.attention_head_size = attention_head_size

            def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                x = x.view(new_x_shape)
                return x.permute(0, 2, 1, 3)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                if self.transpose_for_score:
                    q = self.transpose_for_scores(q)
                    k = self.transpose_for_scores(k)
                    v = self.transpose_for_scores(v)
                scores = torch.matmul(q, k.transpose(-1, -2)) / (self.input_dim**0.5)
                attention = self.softmax(scores)
                weighted = torch.matmul(attention, v)
                return weighted

        for annotate_matmul in [False, True]:
            mod = SelfAttnLikeModule(
                input_dim=64 * 16,
                transpose_for_score=True,
                num_attention_heads=16,
                attention_head_size=64,
            ).eval()
            v = torch.randn(2, 384, 1024)

            def matcher_check_fn():
                self.assertEqual(
                    counters["inductor"]["qlinear_weight_prepack_matcher_count"], 3
                )
                self.assertEqual(
                    counters["inductor"]["qlinear_unary_matcher_count"],
                    3 if annotate_matmul and not TEST_ACL else 0,
                )

            quantizer = X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            if annotate_matmul:
                quantizer.set_function_type_qconfig(
                    torch.matmul, quantizer.get_global_quantization_config()
                )

            self._test_common(
                mod,
                (v,),
                matcher_check_fn,
                check_quantization=True,
                quantizer=quantizer,
            )


instantiate_parametrized_tests(TestPatternMatcher)
if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        # set weight_prepack = False to skip fusion passes in pytorch core
        import torch._inductor.config

        torch._inductor.config.cpp.weight_prepack = False
        run_tests()
