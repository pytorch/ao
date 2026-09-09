# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import copy
import gc
import tempfile
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.testing._internal import common_utils
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import TestCase

from torchao import quantize_
from torchao.float8.config import e4m3_dtype, e5m2_dtype
from torchao.float8.float8_linear import Float8Linear
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
from torchao.quantization import (
    Float8Tensor,
    Int4TilePackedTo4dTensor,
    Int8Tensor,
    IntxUnpackedToInt8Tensor,
    PerGroup,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.qat import (
    FakeQuantizedLinear,
    QATConfig,
)
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Float8WeightOnlyConfig,
    FqnToConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8StaticActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    PerRow,
    PerTensor,
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs
from torchao.quantization.utils import compute_error
from torchao.testing.pt2e._xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.testing.utils import skip_if_rocm, skip_if_xpu
from torchao.utils import (
    get_available_devices,
    get_current_accelerator_device,
    is_ROCM,
    is_sm_at_least_89,
    is_sm_at_least_90,
    is_sm_at_least_100,
    unwrap_tensor_subclass,
)

# Float8 weight-only quantization is not supported on MPS.
REAPPLICATION_DEVICES = [
    device for device in get_available_devices() if device != "mps"
]


def dynamic_quant(model, example_inputs):
    m = torch.export.export(model, example_inputs, strict=True).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_dynamic=True)
    )
    m = prepare_pt2e(m, quantizer)
    m = convert_pt2e(m)
    return m


def capture_and_prepare(model, example_inputs):
    m = torch.export.export(model, example_inputs, strict=True)
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_dynamic=True)
    )
    m = prepare_pt2e(m, quantizer)
    # TODO: we can run the weight observer in convert_pt2e so that user don't need to run this
    m(*example_inputs)
    return m


class XNNPackDynamicQuantizer:
    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_with_custom_fn_if_matches_filter(
            model,
            lambda linear_mod: capture_and_prepare(
                linear_mod, (torch.randn(1, linear_mod.in_features))
            ),
            lambda mod, fqn: isinstance(mod, torch.nn.Linear),
        )
        return model

    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_with_custom_fn_if_matches_filter(
            model,
            lambda linear_mod: convert_pt2e(linear_mod),
            lambda mod, fqn: isinstance(mod, torch.fx.GraphModule),
        )
        return model


class TorchCompileDynamicQuantizer:
    def quantize(self, model: torch.nn.Module) -> torch.nn.Module:
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        return model


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64, bias=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=bias).to(torch.float)
        self.linear2 = torch.nn.Linear(n, k, bias=bias).to(torch.float)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cpu"):
        return (
            torch.randn(
                batch_size, self.linear1.in_features, dtype=dtype, device=device
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def _get_ref_change_linear_weights_to_woqtensors(deprecated_tensor_subclass):
    def _ref_change_linear_weights_to_woqtensors(model, filter_fn=None, **kwargs):
        """
        The deprecated implementation for weight only quant API, used as a reference for
        numerics and performance
        """
        from torchao.quantization.quant_api import _get_subclass_inserter, _is_linear

        filter_fn = kwargs.pop("filter_fn", _is_linear)

        _replace_with_custom_fn_if_matches_filter(
            model,
            _get_subclass_inserter(
                deprecated_tensor_subclass, enable_parametrization=True, **kwargs
            ),
            filter_fn,
        )

    return _ref_change_linear_weights_to_woqtensors


class TestQuantFlow(TestCase):
    GPU_DEVICES = (["cuda"] if torch.cuda.is_available() else []) + (
        ["xpu"] if torch.xpu.is_available() else []
    )

    def setUp(self):
        super().setUp()
        # quantize_() sets a global float32 matmul precision; snapshot fp32_precision
        self._prev_cuda_matmul_fp32 = torch.backends.cuda.matmul.fp32_precision
        self._prev_mkldnn_matmul_fp32 = torch.backends.mkldnn.matmul.fp32_precision

    def tearDown(self):
        torch.backends.cuda.matmul.fp32_precision = self._prev_cuda_matmul_fp32
        torch.backends.mkldnn.matmul.fp32_precision = self._prev_mkldnn_matmul_fp32
        super().tearDown()

    def test_quantize_does_not_leak_fp32_precision(self):
        # quantize_() must not mutate the process wide fp32 precision flags, torch's
        # TestCase.tearDown reports any change as "fp32 precision flag leak detected"
        backends = (
            torch.backends.cuda.matmul,
            torch.backends.cudnn.conv,
            torch.backends.cudnn.rnn,
            torch.backends.mkldnn.matmul,
            torch.backends.mkldnn.conv,
            torch.backends.mkldnn.rnn,
        )
        before = [b.fp32_precision for b in backends]
        m = ToyLinearModel().eval()
        quantize_(m, Int8WeightOnlyConfig())
        self.assertEqual([b.fp32_precision for b in backends], before)

    def _snapshot_reapplication_state(self, model, example_inputs, check_output=True):
        module_states = {}
        for name in ("linear1", "linear2"):
            module = getattr(model, name)
            weight = module.weight
            if hasattr(weight, "__tensor_flatten__"):
                tensor_names, metadata = weight.__tensor_flatten__()
                tensors = {
                    tensor_name: (
                        getattr(weight, tensor_name),
                        getattr(weight, tensor_name).clone(),
                    )
                    for tensor_name in tensor_names
                }
                metadata = copy.deepcopy(metadata)
            else:
                tensor_names = None
                tensors = {"weight": (weight, weight.clone())}
                metadata = None
            module_states[name] = (
                module,
                repr(module),
                weight,
                tensor_names,
                tensors,
                metadata,
            )
        output = model(*example_inputs).clone() if check_output else None
        return module_states, output

    def _assert_reapplication_state_unchanged(
        self, model, example_inputs, expected_state
    ):
        module_states, output = expected_state
        for name, saved_state in module_states.items():
            module, module_repr, weight, tensor_names, tensors, metadata = saved_state
            current_module = getattr(model, name)
            current_weight = current_module.weight
            self.assertIs(current_module, module)
            self.assertEqual(repr(current_module), module_repr)
            self.assertIs(current_weight, weight)
            if tensor_names is not None:
                current_names, current_metadata = current_weight.__tensor_flatten__()
                self.assertEqual(current_names, tensor_names)
                self.assertEqual(current_metadata, metadata)
            for tensor_name, (tensor, tensor_value) in tensors.items():
                current_tensor = (
                    current_weight
                    if tensor_name == "weight"
                    else getattr(current_weight, tensor_name)
                )
                self.assertIs(current_tensor, tensor)
                self.assertEqual(current_tensor, tensor_value)
        if output is not None:
            self.assertEqual(model(*example_inputs), output)

    def _apply_reapplication_config(self, model, config, use_fqn):
        if use_fqn:
            quantize_(
                model,
                FqnToConfig({"linear1.weight": config}),
                filter_fn=None,
            )
        else:
            quantize_(model, config)

    def _get_incompatible_reapplication_configs(self, case, device):
        config_kwargs = {"set_inductor_config": False}
        if case == "int8_row_to_tensor":
            return (
                Int8WeightOnlyConfig(granularity=PerRow(), **config_kwargs),
                Int8WeightOnlyConfig(granularity=PerTensor(), **config_kwargs),
            )
        if case == "int8_tensor_to_row":
            return (
                Int8WeightOnlyConfig(granularity=PerTensor(), **config_kwargs),
                Int8WeightOnlyConfig(granularity=PerRow(), **config_kwargs),
            )
        if case == "float8_row_to_tensor":
            return (
                Float8WeightOnlyConfig(granularity=PerRow(), **config_kwargs),
                Float8WeightOnlyConfig(granularity=PerTensor(), **config_kwargs),
            )
        if case == "float8_tensor_to_row":
            return (
                Float8WeightOnlyConfig(granularity=PerTensor(), **config_kwargs),
                Float8WeightOnlyConfig(granularity=PerRow(), **config_kwargs),
            )
        if case == "int8_to_float8":
            return (
                Int8WeightOnlyConfig(**config_kwargs),
                Float8WeightOnlyConfig(**config_kwargs),
            )
        if case == "float8_to_int8":
            return (
                Float8WeightOnlyConfig(**config_kwargs),
                Int8WeightOnlyConfig(**config_kwargs),
            )
        if case == "int8_group_size":
            return (
                Int8WeightOnlyConfig(granularity=PerGroup(16), **config_kwargs),
                Int8WeightOnlyConfig(granularity=PerGroup(32), **config_kwargs),
            )
        if case == "float8_group_size":
            return (
                Float8WeightOnlyConfig(granularity=PerGroup(16), **config_kwargs),
                Float8WeightOnlyConfig(granularity=PerGroup(32), **config_kwargs),
            )
        if case == "int8_dynamic_to_weight_only":
            return (
                Int8DynamicActivationInt8WeightConfig(**config_kwargs),
                Int8WeightOnlyConfig(**config_kwargs),
            )
        if case == "int8_static_to_weight_only":
            return (
                Int8StaticActivationInt8WeightConfig(
                    act_quant_scale=torch.ones(1, device=device),
                    **config_kwargs,
                ),
                Int8WeightOnlyConfig(**config_kwargs),
            )
        if case == "float8_dtype":
            return (
                Float8WeightOnlyConfig(weight_dtype=e4m3_dtype, **config_kwargs),
                Float8WeightOnlyConfig(weight_dtype=e5m2_dtype, **config_kwargs),
            )
        raise AssertionError(f"Unknown test case: {case}")

    @common_utils.parametrize(
        "config",
        [
            Int8WeightOnlyConfig(set_inductor_config=False),
            Int8WeightOnlyConfig(granularity=PerTensor(), set_inductor_config=False),
            Int8WeightOnlyConfig(granularity=PerGroup(16), set_inductor_config=False),
            Float8WeightOnlyConfig(set_inductor_config=False),
            Float8WeightOnlyConfig(granularity=PerTensor(), set_inductor_config=False),
            Float8WeightOnlyConfig(granularity=PerGroup(16), set_inductor_config=False),
        ],
    )
    @common_utils.parametrize("device", REAPPLICATION_DEVICES)
    def test_reapply_weight_only_config_is_noop(self, config, device):
        torch.manual_seed(0)
        model = ToyLinearModel().eval().to(device)
        example_inputs = model.example_inputs(device=device)

        quantize_(model, config)
        expected_state = self._snapshot_reapplication_state(model, example_inputs)

        # Reproduce a caller quantizing a container and then one of its children.
        quantize_(model.linear1, config)
        self._assert_reapplication_state_unchanged(
            model, example_inputs, expected_state
        )

        # A fresh equivalent config must also leave the model unchanged.
        quantize_(model, copy.deepcopy(config))
        self._assert_reapplication_state_unchanged(
            model, example_inputs, expected_state
        )

    @common_utils.parametrize(
        "config",
        [
            Int8WeightOnlyConfig(set_inductor_config=False),
            Int8WeightOnlyConfig(granularity=PerTensor(), set_inductor_config=False),
            Int8WeightOnlyConfig(granularity=PerGroup(16), set_inductor_config=False),
            Float8WeightOnlyConfig(set_inductor_config=False),
            Float8WeightOnlyConfig(granularity=PerTensor(), set_inductor_config=False),
            Float8WeightOnlyConfig(granularity=PerGroup(16), set_inductor_config=False),
        ],
    )
    @common_utils.parametrize("fqn", ["linear1", "linear1.weight"])
    @common_utils.parametrize("device", REAPPLICATION_DEVICES)
    def test_reapply_weight_only_fqn_config_is_noop(self, config, fqn, device):
        torch.manual_seed(0)
        model = ToyLinearModel().eval().to(device)
        example_inputs = model.example_inputs(device=device)
        fqn_config = FqnToConfig({fqn: config})

        quantize_(model, fqn_config, filter_fn=None)
        expected_state = self._snapshot_reapplication_state(model, example_inputs)

        quantize_(model, fqn_config, filter_fn=None)
        self._assert_reapplication_state_unchanged(
            model, example_inputs, expected_state
        )

        fresh_fqn_config = FqnToConfig({fqn: copy.deepcopy(config)})
        quantize_(model, fresh_fqn_config, filter_fn=None)
        self._assert_reapplication_state_unchanged(
            model, example_inputs, expected_state
        )

    @common_utils.parametrize(
        "config",
        [
            Int8WeightOnlyConfig(set_inductor_config=False),
            Float8WeightOnlyConfig(set_inductor_config=False),
        ],
    )
    def test_reapply_weight_only_config_is_noop_on_meta(self, config):
        model = torch.nn.Sequential(torch.nn.Linear(16, 16, device="meta"))
        quantize_(model, config)
        expected_module = model[0]
        expected_weight = model[0].weight
        tensor_names, expected_metadata = expected_weight.__tensor_flatten__()
        expected_tensors = {
            name: getattr(expected_weight, name) for name in tensor_names
        }

        quantize_(model, copy.deepcopy(config))

        self.assertIs(model[0], expected_module)
        self.assertIs(model[0].weight, expected_weight)
        current_names, current_metadata = model[0].weight.__tensor_flatten__()
        self.assertEqual(current_names, tensor_names)
        self.assertEqual(current_metadata, expected_metadata)
        for name, tensor in expected_tensors.items():
            self.assertIs(getattr(model[0].weight, name), tensor)

    def test_reapply_int8_weight_only_config_is_noop_on_fake_tensor(self):
        with FakeTensorMode():
            model = torch.nn.Sequential(torch.nn.Linear(16, 16))
            config = Int8WeightOnlyConfig(set_inductor_config=False)
            quantize_(model, config)
            expected_module = model[0]
            expected_weight = model[0].weight
            self.assertTrue(is_fake(expected_weight.zero_point))

            quantize_(model, copy.deepcopy(config))

            self.assertIs(model[0], expected_module)
            self.assertIs(model[0].weight, expected_weight)

    @common_utils.parametrize("use_fqn", [False, True])
    @common_utils.parametrize(
        "case",
        [
            "int8_row_to_tensor",
            "int8_tensor_to_row",
            "float8_row_to_tensor",
            "float8_tensor_to_row",
            "int8_to_float8",
            "float8_to_int8",
            "int8_group_size",
            "float8_group_size",
            "int8_dynamic_to_weight_only",
            "int8_static_to_weight_only",
            "float8_dynamic_to_weight_only",
            "float8_dtype",
        ],
    )
    @common_utils.parametrize("device", REAPPLICATION_DEVICES)
    def test_weight_only_reapplication_rejects_incompatible_representation(
        self, use_fqn, case, device
    ):
        torch.manual_seed(0)
        model = ToyLinearModel().eval().to(device)
        example_inputs = model.example_inputs(device=device)
        if case == "float8_dynamic_to_weight_only":
            # Construct the representation directly so this negative test does not
            # depend on Float8 dynamic hardware support.
            dynamic_config = Float8DynamicActivationFloat8WeightConfig(
                set_inductor_config=False
            )
            act_granularity, weight_granularity = dynamic_config.granularity
            quantized_weight = Float8Tensor.from_hp(
                model.linear1.weight.detach(),
                float8_dtype=dynamic_config.weight_dtype,
                granularity=weight_granularity,
                mm_config=dynamic_config.mm_config,
                kernel_preference=dynamic_config.kernel_preference,
                act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
                    float8_dtype=dynamic_config.activation_dtype,
                    granularity=act_granularity,
                    hp_value_lb=dynamic_config.activation_value_lb,
                    hp_value_ub=dynamic_config.activation_value_ub,
                    kernel_preference=dynamic_config.kernel_preference,
                ),
            )
            model.linear1.weight = torch.nn.Parameter(
                quantized_weight, requires_grad=False
            )
            second_config = Float8WeightOnlyConfig(set_inductor_config=False)
        else:
            first_config, second_config = self._get_incompatible_reapplication_configs(
                case, device
            )
            self._apply_reapplication_config(model, first_config, use_fqn)

        check_output = case not in {
            "int8_static_to_weight_only",
            "float8_dynamic_to_weight_only",
        }
        expected_state = self._snapshot_reapplication_state(
            model, example_inputs, check_output=check_output
        )

        with self.assertRaisesRegex(ValueError, "is not equivalent to the existing"):
            self._apply_reapplication_config(model, second_config, use_fqn)

        self._assert_reapplication_state_unchanged(
            model, example_inputs, expected_state
        )

    @common_utils.parametrize("case", ["int8_asymmetric", "float8_missing_block_size"])
    @common_utils.parametrize("device", REAPPLICATION_DEVICES)
    def test_weight_only_reapplication_rejects_incompatible_tensor_metadata(
        self, case, device
    ):
        model = ToyLinearModel().eval().to(device)
        weight = model.linear1.weight.detach()
        if case == "int8_asymmetric":
            quantized_weight = Int8Tensor.from_hp(
                weight,
                granularity=PerRow(),
                mapping_type=MappingType.ASYMMETRIC,
            )
            config = Int8WeightOnlyConfig(set_inductor_config=False)
        else:
            quantized_weight = Float8Tensor(
                weight.to(e4m3_dtype),
                torch.ones(1, dtype=torch.float32, device=device),
                dtype=weight.dtype,
            )
            config = Float8WeightOnlyConfig(set_inductor_config=False)
        model.linear1.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
        example_inputs = model.example_inputs(device=device)
        expected_state = self._snapshot_reapplication_state(
            model, example_inputs, check_output=False
        )

        with self.assertRaisesRegex(ValueError, "is not equivalent to the existing"):
            quantize_(model.linear1, config)

        self._assert_reapplication_state_unchanged(
            model, example_inputs, expected_state
        )

    def test_reapply_float8_weight_only_unwraps_float8_linear(self):
        float8_linear = Float8Linear.from_float(torch.nn.Linear(4, 4).eval()).eval()
        float8_linear.weight = torch.nn.Parameter(
            Float8Tensor.from_hp(
                float8_linear.weight.detach(), float8_dtype=e4m3_dtype
            ),
            requires_grad=False,
        )
        model = torch.nn.Sequential(float8_linear)
        expected_weight = model[0].weight
        expected_bias = model[0].bias
        example_input = torch.randn(2, 4)
        expected_output = torch.nn.functional.linear(
            example_input, expected_weight, expected_bias
        )

        quantize_(
            model,
            Float8WeightOnlyConfig(set_inductor_config=False),
        )

        self.assertIs(type(model[0]), torch.nn.Linear)
        self.assertIs(model[0].weight, expected_weight)
        self.assertIs(model[0].bias, expected_bias)
        self.assertFalse(model[0].training)
        self.assertEqual(model(example_input), expected_output)

    @common_utils.parametrize(
        "config",
        [
            Int8WeightOnlyConfig(set_inductor_config=False),
            Float8WeightOnlyConfig(set_inductor_config=False),
        ],
    )
    def test_reapply_weight_only_config_sets_inductor_config(self, config):
        model = ToyLinearModel().eval()
        inductor_settings = {
            "coordinate_descent_tuning": False,
            "coordinate_descent_check_all_directions": False,
            "force_fuse_int_mm_with_mul": False,
            "fx_graph_cache": False,
            "triton.unique_kernel_names": False,
        }

        with torch._inductor.config.patch(inductor_settings):
            quantize_(model, config)
            second_config = copy.deepcopy(config)
            second_config.set_inductor_config = True
            quantize_(model, second_config)

            self.assertTrue(torch._inductor.config.coordinate_descent_tuning)
            self.assertTrue(
                torch._inductor.config.coordinate_descent_check_all_directions
            )
            self.assertTrue(torch._inductor.config.force_fuse_int_mm_with_mul)
            self.assertTrue(torch._inductor.config.fx_graph_cache)
            self.assertTrue(torch._inductor.config.triton.unique_kernel_names)

    def test_dynamic_quant_gpu_singleline(self):
        if is_ROCM():
            self.skipTest("Don't test CPU for ROCM version of torch")
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        m(*example_inputs)
        # AssertionError: Expecting input to have dtype torch.float32, but got dtype: torch.float64
        # While executing %choose_qparams_tensor_1 : [num_users=2] = call_function[target=torch.ops.quantized_decomposed.choose_qparams.tensor](args = (%arg0_3, -128, 127, 0.000244140625, torch.int8), kwargs = {})
        # m = torch.compile(m, mode="max-autotune")
        # print(example_inputs[0].dtype)
        # compiled = m(*example_inputs)
        # torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    @unittest.skip("skipping for now due to torch.compile error")
    def test_dynamic_quant_gpu_unified_api_unified_impl(self):
        quantizer = XNNPackDynamicQuantizer()
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        m = quantizer.prepare(m)
        m = quantizer.convert(m)
        quantized = m(*example_inputs)
        # AssertionError: Expecting input to have dtype torch.float32, but got dtype: torch.float64
        # While executing %choose_qparams_tensor_1 : [num_users=2] = call_function[target=torch.ops.quantized_decomposed.choose_qparams.tensor](args = (%arg0_3, -128, 127, 0.000244140625, torch.int8), kwargs = {})
        m = torch.compile(m, mode="max-autotune")
        # print(example_inputs[0].dtype)
        compiled = m(*example_inputs)
        torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    @unittest.skip(
        "FAILED test/quantization/test_quant_api.py::TestQuantFlow::test_dynamic_quant_gpu_unified_api_eager_mode_impl - AssertionError: Tensor-likes are not equal!"
    )
    def test_dynamic_quant_gpu_unified_api_eager_mode_impl(self):
        quantizer = TorchCompileDynamicQuantizer()
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        quantized = m(*example_inputs)
        m = torch.compile(m, mode="max-autotune")
        compiled = m(*example_inputs)
        torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_int8_wo_quant_save_load(self):
        m = ToyLinearModel().eval().cpu()

        def api(model):
            quantize_(model, Int8WeightOnlyConfig())
            unwrap_tensor_subclass(model)

        api(m)

        example_inputs = m.example_inputs()
        ref = m(*example_inputs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(m.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)

        m2 = ToyLinearModel().eval().cpu()
        api(m2)

        m2.load_state_dict(state_dict)
        device = get_current_accelerator_device()
        m2 = m2.to(device)
        example_inputs = map(lambda x: x.to(device), example_inputs)
        res = m2(*example_inputs)

        # TODO: figure out why ROCm has a larger error
        atol, rtol = (1e-2, 1e-2) if torch.version.hip else (None, None)
        torch.testing.assert_close(ref, res.cpu(), atol=atol, rtol=rtol)

    def test_8da4w_quantizer(self):
        from torchao.quantization.linear_quant_modules import Int8DynActInt4WeightLinear
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        quantizer = Int8DynActInt4WeightQuantizer(groupsize=32)
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        assert isinstance(m.linear1, Int8DynActInt4WeightLinear)
        assert isinstance(m.linear2, Int8DynActInt4WeightLinear)
        m(*example_inputs)

    def test_8da4w_quantizer_linear_bias(self):
        from torchao.quantization.linear_quant_modules import Int8DynActInt4WeightLinear
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        quantizer = Int8DynActInt4WeightQuantizer(groupsize=32)
        m = ToyLinearModel(bias=True).eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        assert isinstance(m.linear1, Int8DynActInt4WeightLinear)
        assert isinstance(m.linear2, Int8DynActInt4WeightLinear)
        m(*example_inputs)

    def test_8da4w_quantizer_scales_precision(self):
        # Regression test: _convert_for_runtime must build the runtime modules
        # with scales_precision (not precision) for their scales/zeros buffers.
        # It previously passed self.precision, so a quantizer configured with a
        # distinct scales_precision silently produced runtime modules whose
        # scales were cast to the activation precision instead.
        from torchao.quantization.linear_quant_modules import Int8DynActInt4WeightLinear
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        quantizer = Int8DynActInt4WeightQuantizer(
            groupsize=32,
            precision=torch.float32,
            scales_precision=torch.bfloat16,
        )
        m = ToyLinearModel(bias=True).eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        for name in ("linear1", "linear2"):
            mod = getattr(m, name)
            assert isinstance(mod, Int8DynActInt4WeightLinear)
            # scales/zeros buffers follow scales_precision...
            self.assertEqual(mod.scales.dtype, torch.bfloat16)
            self.assertEqual(mod.zeros.dtype, torch.bfloat16)
            # ...while the activation precision and bias buffer still follow
            # precision. Asserting these guards against a future change that
            # over-corrects by routing everything through scales_precision.
            self.assertEqual(mod.precision, torch.float32)
            self.assertEqual(mod.bias.dtype, torch.float32)
        m(*example_inputs)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_quantized_tensor_subclass_save_load(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        m_copy = copy.deepcopy(m)
        example_inputs = m.example_inputs(dtype=torch.bfloat16)

        quantize_(m, Int8WeightOnlyConfig())
        ref = m(*example_inputs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(m.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)

        m_copy.load_state_dict(state_dict, assign=True)

        res = m_copy(*example_inputs)
        self.assertEqual(res, ref)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_int8wo_quantized_model_to_device(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        example_inputs = m.example_inputs(dtype=torch.bfloat16, device="cpu")

        quantize_(m, Int8WeightOnlyConfig())
        ref = m(*example_inputs)

        device = get_current_accelerator_device()
        example_inputs_cuda = (example_inputs[0].to(device),)
        m.to(device)
        cuda_res = m(*example_inputs_cuda)
        self.assertEqual(cuda_res.cpu(), ref)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_quantized_tensor_subclass_save_load_map_location(self):
        device = get_current_accelerator_device()
        m = ToyLinearModel().eval().to(dtype=torch.bfloat16, device=device)
        example_inputs = m.example_inputs(dtype=torch.bfloat16, device=device)

        quantize_(m, Int8WeightOnlyConfig())
        ref = m(*example_inputs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(m.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f.name, map_location="cpu", mmap=True)

        with torch.device("meta"):
            m_copy = ToyLinearModel().eval()

        m_copy.load_state_dict(state_dict, assign=True)
        m_copy.to(dtype=torch.bfloat16, device=device)

        res = m_copy(*example_inputs)
        self.assertEqual(res, ref)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_quantized_model_streaming(self):
        device = get_current_accelerator_device()
        device_module = torch.get_device_module(device)

        def reset_memory():
            gc.collect()
            device_module.empty_cache()
            device_module.reset_peak_memory_stats()

        reset_memory()
        m = ToyLinearModel()
        quantize_(m.to(device=device), Int8WeightOnlyConfig())
        memory_baseline = device_module.max_memory_allocated()

        del m
        reset_memory()
        m = ToyLinearModel()
        quantize_(m, Int8WeightOnlyConfig(), device=device)
        memory_streaming = device_module.max_memory_allocated()

        for param in m.parameters():
            assert param.device.type == device.type
        self.assertLess(memory_streaming, memory_baseline)

    # TODO(#1690): move to new config names
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @common_utils.parametrize(
        "config",
        [
            Float8WeightOnlyConfig(),
            Float8DynamicActivationFloat8WeightConfig(),
            Int8DynamicActivationInt8WeightConfig(),
            Int8WeightOnlyConfig(),
        ],
    )
    @skip_if_xpu("XPU enablement in progress")
    @skip_if_rocm("ROCm enablement in progress")
    def test_workflow_e2e_numerics(self, config):
        """
        Simple test of e2e Int4WeightOnlyConfig workflow, comparing numerics
        to a bfloat16 baseline.
        """
        if (
            isinstance(
                config,
                Float8DynamicActivationFloat8WeightConfig,
            )
            and not is_sm_at_least_89()
        ):
            return unittest.skip("requires CUDA capability 8.9 or greater")

        dtype = torch.bfloat16

        # set up inputs
        device = get_current_accelerator_device()
        x = torch.randn(128, 128, device=device, dtype=dtype)
        # TODO(future): model in float32 leads to error: https://gist.github.com/vkuzo/63b3bcd7818393021a6e3fb4ccf3c469
        # is that expected?
        m_ref = torch.nn.Sequential(torch.nn.Linear(128, 128)).to(device).to(dtype)
        m_q = copy.deepcopy(m_ref)

        # quantize
        quantize_(m_q, config)

        with torch.no_grad():
            y_ref = m_ref(x)
            y_q = m_q(x)

        sqnr = compute_error(y_ref, y_q)
        assert sqnr >= 16.5, f"SQNR {sqnr} is too low"

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(not is_sm_at_least_89(), "Need SM 8.9+")
    def test_module_fqn_to_config_default(self):
        config1 = Float8DynamicActivationFloat8WeightConfig()
        config2 = Int8WeightOnlyConfig()
        config = ModuleFqnToConfig({"_default": config1, "linear2": config2})
        device = get_current_accelerator_device()
        model = ToyLinearModel().to(device).to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device=device, dtype=torch.bfloat16)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert isinstance(model.linear2.weight, IntxUnpackedToInt8Tensor)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(not is_sm_at_least_89(), "Need SM 8.9+")
    def test_module_fqn_to_config_module_name(self):
        config1 = Float8DynamicActivationFloat8WeightConfig()
        config2 = Int8WeightOnlyConfig()
        config = ModuleFqnToConfig({"linear1": config1, "linear2": config2})
        device = get_current_accelerator_device()
        model = ToyLinearModel().to(device).to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device=device, dtype=torch.bfloat16)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert isinstance(model.linear2.weight, IntxUnpackedToInt8Tensor)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_regex_basic(self):
        config1 = Int4WeightOnlyConfig(
            group_size=32, int4_packing_format="tile_packed_to_4d"
        )
        config = ModuleFqnToConfig({"re:linear.": config1})
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Int4TilePackedTo4dTensor)
        assert isinstance(model.linear2.weight, Int4TilePackedTo4dTensor)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_regex_precedence(self):
        """Testing that full path config takes precedence over
        regex config in ModuleFqnToConfig
        """
        config1 = Int4WeightOnlyConfig(
            group_size=32, int4_packing_format="tile_packed_to_4d"
        )
        config2 = IntxWeightOnlyConfig()
        config = ModuleFqnToConfig({"linear1": config1, "re:linear.": config2})
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Int4TilePackedTo4dTensor)
        assert isinstance(model.linear2.weight, IntxUnpackedToInt8Tensor)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_regex_precedence2(self):
        """Testing that full path config takes precedence over
        regex config in ModuleFqnToConfig, swapping
        the order of `re:linear.*` and `linear1` to make sure that
        `linear1` config has precedence even it comes after `linear*`
        """
        config1 = Int4WeightOnlyConfig(
            group_size=32, int4_packing_format="tile_packed_to_4d"
        )
        config2 = IntxWeightOnlyConfig()
        config = ModuleFqnToConfig({"re:linear.": config2, "linear1": config1})
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Int4TilePackedTo4dTensor)
        assert isinstance(model.linear2.weight, IntxUnpackedToInt8Tensor)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_regex_fullmatch(self):
        """Testing that we will only match the fqns that fully
        matches the regex
        """

        class M(torch.nn.Module):
            def __init__(self, dtype, device):
                super().__init__()
                self.dtype = dtype
                self.device = device
                self.linear1 = torch.nn.Linear(32, 64, dtype=dtype, device=device)
                self.not_full_match_linear2 = torch.nn.Linear(
                    64, 32, dtype=dtype, device=device
                )
                self.linear3_full_match = torch.nn.Linear(
                    32, 32, dtype=dtype, device=device
                )

            def forward(self, x):
                x = self.linear1(x)
                x = self.not_full_match_linear2(x)
                x = self.linear3_full_match(x)
                return

            def example_inputs(self):
                return (torch.randn(1, 32, dtype=self.dtype, device=self.device),)

        config1 = Int4WeightOnlyConfig(
            group_size=32, int4_packing_format="tile_packed_to_4d"
        )
        config2 = Float8WeightOnlyConfig()
        config = ModuleFqnToConfig(
            {
                "re:linear.*": config2,
                "linear1": config1,
                "linear3_full_match.bias": None,
            }
        )
        model = M(dtype=torch.bfloat16, device="cuda")
        example_inputs = model.example_inputs()
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Int4TilePackedTo4dTensor)
        # since fqn does not fully match `linear*`, it should not be quantized
        assert not isinstance(model.not_full_match_linear2.weight, Float8Tensor)
        # linear3_full_match matches `linear*`, so should be quantized
        assert isinstance(model.linear3_full_match.weight, Float8Tensor)

    def test_module_fqn_to_config_embedding_linear(self):
        weight_dtype = torch.int8
        granularity = PerGroup(8)
        mapping_type = MappingType.SYMMETRIC
        embedding_config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=mapping_type,
        )
        # example model linear is Linear(16, 8)
        linear_config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(16),
        )

        config = ModuleFqnToConfig({"emb": embedding_config, "linear": linear_config})
        indices = torch.randint(0, 10, (32,))
        indices = indices.unsqueeze(0)
        example_inputs = (indices,)
        model = TestHelperModules.EmbeddingConvLinearModule().eval()
        model(*example_inputs)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)

        assert isinstance(model.emb.weight, IntxUnpackedToInt8Tensor)
        assert isinstance(model.linear.weight, IntxUnpackedToInt8Tensor)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(not is_sm_at_least_89(), "Need SM 8.9+")
    def test_module_fqn_to_config_skip(self):
        config1 = Float8DynamicActivationFloat8WeightConfig()
        config = ModuleFqnToConfig({"_default": config1, "linear2": None})
        device = get_current_accelerator_device()
        model = ToyLinearModel().to(device).to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device=device, dtype=torch.bfloat16)
        quantize_(model, config, filter_fn=None)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert not isinstance(model.linear2.weight, Float8Tensor)


common_utils.instantiate_parametrized_tests(TestQuantFlow)


@unittest.skipIf(not torch.accelerator.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Checkpoints are produced in SM90+")
class TestFqnToConfig(TestCase):
    def test_fqn_to_config_repr_custom(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "x", torch.nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16))
                )
                self.register_parameter(
                    "y", torch.nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16))
                )

        custom_module = TestModule().cuda().eval()
        custom_module_config = FqnToConfig(
            {
                "x": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
            }
        )
        quantize_(
            custom_module,
            custom_module_config,
            filter_fn=None,
        )
        assert str(custom_module).startswith("TestModule(x=Float8Tensor(")
        # Check that the quantization type info (without full tensor data) is in the module repr
        assert "Float8Tensor(" in str(custom_module)
        assert "PerTensor()" in str(custom_module)

    def test_fqn_to_config_repr_linear(self):
        linear_model = ToyLinearModel().to(torch.bfloat16).cuda().eval()
        linear_quant_config = FqnToConfig(
            {
                "linear1.weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
            }
        )
        quantize_(
            linear_model,
            linear_quant_config,
            filter_fn=None,
        )
        expected_starting_str = (
            "Linear(in_features=64, out_features=32, bias=False, weight=Float8Tensor("
        )

        assert str(linear_model.linear1).startswith(expected_starting_str)
        # Check that the quantization type info (without full tensor data) is in the module repr
        assert "Float8Tensor(" in str(linear_model)
        assert "PerTensor()" in str(linear_model)

    def test_fqn_to_config_regex_skip(self):
        """Test that regex pattern with None config skips matching modules."""

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embed = torch.nn.Sequential(
                    torch.nn.Linear(128, 128), torch.nn.Linear(128, 128)
                )
                self.linear1 = torch.nn.Linear(128, 128)

            def forward(self, x):
                x = self.time_embed(x)
                x = self.linear1(x)
                return x

        model = TestModel().eval()

        cfg = FqnToConfig(
            {
                "re:.*time_embed.*": None,
                "_default": Float8WeightOnlyConfig(),
            }
        )

        quantize_(model, cfg, filter_fn=None)

        # time_embed linears should NOT be quantized (regex matched with None)
        for name, mod in model.time_embed.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not isinstance(mod.weight, Float8Tensor), (
                    f"time_embed.{name}.weight should not be quantized"
                )

        # linear1 should be quantized via _default
        assert isinstance(model.linear1.weight, Float8Tensor)

    def test_quantize_param_fqn_exact(self):
        from transformers import AutoConfig
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        config = AutoConfig.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        ).text_config
        device = get_current_accelerator_device()
        model = Llama4TextMoe(config).to(torch.bfloat16).to(device)

        quant_config = FqnToConfig(
            {
                "experts.gate_up_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
            filter_fn=None,
        )

        assert isinstance(model.experts.gate_up_proj, Float8Tensor)

    def test_quantize_param_fqn_regex(self):
        from transformers import AutoConfig
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

        config = AutoConfig.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        ).text_config
        model = Llama4TextMoe(config).to(torch.bfloat16).cuda()

        quant_config = FqnToConfig(
            {
                "re:.*gate_up_proj": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
            filter_fn=None,
        )

        assert isinstance(model.experts.gate_up_proj, Float8Tensor)

    def test_quantize_fqn_precedence_param_over_module(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()

        quant_config = FqnToConfig(
            {
                "linear1": None,
                "linear1.weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                ),
            }
        )
        quantize_(model, quant_config, filter_fn=None)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1

    def test_quantize_fqn_precedence_param_over_module_regex(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()

        quant_config = FqnToConfig(
            {
                "re:linear.*": None,
                "linear1.weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                ),
            }
        )
        quantize_(model, quant_config, filter_fn=None)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1

    def test_quantize_fqn_precedence_param_regex_over_module_regex(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()

        quant_config = FqnToConfig(
            {
                "re:linear.*": None,
                "re:linear.*.weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                ),
            }
        )
        quantize_(model, quant_config, filter_fn=None)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1

    def test_quantize_fqn_precedence_module_over_param_regex(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()

        quant_config = FqnToConfig(
            {
                "re:linear.*.weight": None,
                "linear1": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                ),
            }
        )
        quantize_(model, quant_config, filter_fn=None)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1
        assert not isinstance(model.linear2.weight, Float8Tensor)

    def test_quantize_fqn_precedence_param_over_default(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()

        quant_config = FqnToConfig(
            {
                "linear2.weight": None,
                "_default": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                ),
            }
        )
        quantize_(model, quant_config, filter_fn=None)
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1
        assert not isinstance(model.linear2.weight, Float8Tensor)

    def test_quantize_fqn_precedence_param_regex_over_default(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()

        quant_config = FqnToConfig(
            {
                "re:linear.*.weight": None,
                "_default": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                ),
            }
        )
        quantize_(model, quant_config, filter_fn=None)
        assert not isinstance(model.linear2.weight, Float8Tensor)
        assert not isinstance(model.linear1.weight, Float8Tensor)

    def test_quantize_model_same_module_different_param(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()
        model.linear1.register_parameter(
            "weight2", torch.nn.Parameter(model.linear1.weight.clone())
        )
        quant_config = FqnToConfig(
            {
                "linear1.weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
                "linear1.weight2": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                ),
            }
        )

        quantize_(
            model,
            quant_config,
            filter_fn=None,
        )
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1
        assert isinstance(model.linear1.weight2, Float8Tensor)
        assert model.linear1.weight2.scale.numel() == 32

    def test_quantize_model_same_module_different_param_regex(self):
        model = ToyLinearModel().to(torch.bfloat16).cuda().eval()
        quant_config = FqnToConfig(
            {
                "re:.*weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor(),
                ),
                "re:.*bias": None,
            }
        )

        quantize_(
            model,
            quant_config,
            filter_fn=None,
        )
        assert isinstance(model.linear1.weight, Float8Tensor)
        assert model.linear1.weight.scale.numel() == 1
        assert not isinstance(model.linear1.bias, Float8Tensor)
        assert isinstance(model.linear2.weight, Float8Tensor)
        assert model.linear2.weight.scale.numel() == 1
        assert not isinstance(model.linear2.bias, Float8Tensor)

    def test_unsupported_param_config_raises_not_implemented_error(self):
        """Test that using an unsupported parameter config raises NotImplementedError.

        This test creates a custom config whose handler does not have a 'parameter_name'
        kwarg in its signature. This verifies that _handler_supports_fqn_quantization()
        correctly identifies handlers that don't support parameter-level quantization.
        """
        from dataclasses import dataclass

        from torchao.core.config import AOBaseConfig
        from torchao.quantization.transform_module import (
            register_quantize_module_handler,
        )

        # Create a custom config that doesn't support parameter quantization
        @dataclass
        class TestUnsupportedParamConfig(AOBaseConfig):
            dummy: int = 1

        # Register a handler WITHOUT parameter_name kwarg
        @register_quantize_module_handler(TestUnsupportedParamConfig)
        def _test_unsupported_param_transform(
            module: torch.nn.Module,
            config: TestUnsupportedParamConfig,
        ) -> torch.nn.Module:
            # This handler doesn't have parameter_name, so it can't support param quantization
            return module

        # Create a simple model
        model = torch.nn.Sequential(torch.nn.Linear(10, 5).cuda().bfloat16())

        # Create config targeting a parameter (not a module)
        quant_config = FqnToConfig(
            {
                "0.weight": TestUnsupportedParamConfig(),
            }
        )

        # This should raise NotImplementedError because the handler
        # does not have 'parameter_name' in its signature
        with self.assertRaises(NotImplementedError) as cm:
            quantize_(model, quant_config, filter_fn=None)

        self.assertIn("does not yet support parameter quantization", str(cm.exception))

    def test_filter_fn_and_fqn_to_config_error(self):
        """Test that specifying non-default filter_fn and FqnToConfig raises ValueError."""

        # Create a simple model
        model = torch.nn.Sequential(torch.nn.Linear(10, 5).cuda().bfloat16())

        # Create config with unsupported parameter handler
        quant_config = FqnToConfig(
            {
                "0.weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                )
            }
        )

        # This should raise ValueError
        with self.assertRaises(ValueError):
            quantize_(model, quant_config, filter_fn=lambda mod, fqn: True)

    def test_top_level_param(self):
        model = torch.nn.Linear(16, 16).cuda().bfloat16()

        quant_config = FqnToConfig(
            {
                "weight": Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerTensor()
                )
            }
        )

        quantize_(model, quant_config, filter_fn=None)

        assert isinstance(model.weight, Float8Tensor)
        assert model.weight.scale.numel() == 1

    def test_non_fqn_config_filter_fn_none(self):
        model = torch.nn.Linear(16, 16).cuda().bfloat16()
        quant_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor()
        )

        quantize_(model, quant_config, filter_fn=None)
        assert isinstance(model.weight, Float8Tensor)
        assert model.weight.scale.numel() == 1

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_quantized_model_streaming_fqn_config(self):
        device = get_current_accelerator_device()
        device_module = torch.get_device_module(device)

        def reset_memory():
            gc.collect()
            device_module.empty_cache()
            device_module.reset_peak_memory_stats()

        quant_config = FqnToConfig({"_default": Int8WeightOnlyConfig()})
        reset_memory()
        m = ToyLinearModel()
        quantize_(m.to(device=device), quant_config, filter_fn=None)
        memory_baseline = device_module.max_memory_allocated()

        del m
        reset_memory()
        m = ToyLinearModel()
        quantize_(m, quant_config, device=device, filter_fn=None)
        memory_streaming = device_module.max_memory_allocated()

        for param in m.parameters():
            assert param.device.type == device.type
        self.assertLess(memory_streaming, memory_baseline)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_fqn_config_quantized_nested_module(self):
        class NestedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

        class TopLevelModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = NestedModule()
                self.linear1 = torch.nn.Linear(16, 16)

        m = TopLevelModule()
        quant_config = FqnToConfig(
            {
                "nested.linear": Int8WeightOnlyConfig(),
                "linear1": Int8WeightOnlyConfig(),
            }
        )
        quantize_(m, quant_config, filter_fn=None)

        assert isinstance(m.nested.linear.weight, Int8Tensor)
        assert isinstance(m.linear1.weight, Int8Tensor)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_fqn_config_quantized_nested_module_module_swap(self):
        class NestedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

        class TopLevelModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = NestedModule()
                self.linear1 = torch.nn.Linear(16, 16)

        m = TopLevelModule()
        config = QATConfig(Int4WeightOnlyConfig(), step="prepare")
        quant_config = FqnToConfig(
            {
                "nested.linear": config,
                "linear1": config,
            }
        )
        quantize_(m, quant_config, filter_fn=None)

        assert isinstance(m.nested.linear, FakeQuantizedLinear)
        assert isinstance(m.linear1, FakeQuantizedLinear)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    def test_fqn_config_quantized_nested_module_param(self):
        class NestedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

        class TopLevelModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.nested = NestedModule()
                self.linear1 = torch.nn.Linear(16, 16)

        m = TopLevelModule()
        quant_config = FqnToConfig(
            {
                "nested.linear.weight": Int8WeightOnlyConfig(),
                "linear1.weight": Int8WeightOnlyConfig(),
            }
        )
        quantize_(m, quant_config, filter_fn=None)

        assert isinstance(m.nested.linear.weight, Int8Tensor)
        assert isinstance(m.linear1.weight, Int8Tensor)

    def test_fqn_to_config_non_weight_param(self):
        configs = [
            Int4WeightOnlyConfig(group_size=128),
            Float8DynamicActivationInt4WeightConfig(),
            Int8WeightOnlyConfig(),
            Int8DynamicActivationInt8WeightConfig(),
            Int8DynamicActivationIntxWeightConfig(),
            Int8StaticActivationInt8WeightConfig(),
            IntxWeightOnlyConfig(),
            Float8WeightOnlyConfig(),
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        ]
        if is_sm_at_least_100():
            configs.append(MXDynamicActivationMXWeightConfig())
        if is_sm_at_least_100():
            configs.append(NVFP4DynamicActivationNVFP4WeightConfig())
        for config in configs:
            with self.subTest(config=type(config).__name__):
                model = torch.nn.Sequential(
                    torch.nn.Linear(128, 128).to(torch.bfloat16).cuda()
                )
                model[0].register_parameter(
                    "custom_param",
                    torch.nn.Parameter(
                        torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
                    ),
                )
                original_custom_param = model[0].custom_param
                original_weight = model[0].weight
                quant_config = FqnToConfig({"0.custom_param": config})
                quantize_(model, quant_config, filter_fn=None)
                assert model[0].custom_param is not original_custom_param, (
                    f"custom_param should be quantized for {type(config).__name__}"
                )
                assert model[0].weight is original_weight, (
                    f"weight should be unchanged for {type(config).__name__}"
                )

    def test_fqn_config_module_config_and_fqn_config_both_specified(self):
        with self.assertRaises(ValueError):
            FqnToConfig(
                fqn_to_config={"test": Float8WeightOnlyConfig()},
                module_fqn_to_config={"test2": Float8WeightOnlyConfig()},
            )


if __name__ == "__main__":
    unittest.main()
