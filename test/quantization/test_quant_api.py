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
from pathlib import Path

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import TestCase

from torchao import quantize_
from torchao._models.llama.model import Transformer, prepare_inputs_for_model
from torchao._models.llama.tokenizer import get_tokenizer
from torchao.dtypes import (
    AffineQuantizedTensor,
    Int4CPULayout,
    Int4XPULayout,
    PlainLayout,
    QDQLayout,
    TensorCoreTiledLayout,
)
from torchao.quantization import (
    LinearActivationQuantizedTensor,
    PerGroup,
)
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    Quantizer,
    TwoStepQuantizer,
    _replace_with_custom_fn_if_matches_filter,
    float8_dynamic_activation_float8_weight,
    float8_static_activation_float8_weight,
    float8_weight_only,
    fpx_weight_only,
    gemlite_uintx_weight_only,
    int4_dynamic_activation_int4_weight,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    uintx_weight_only,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.subclass import (
    Int4WeightOnlyQuantizedLinearWeight,
    Int8WeightOnlyQuantizedLinearWeight,
)
from torchao.quantization.utils import compute_error
from torchao.testing.model_architectures import ToyLinearModel
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_3,
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_6,
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_89,
    is_sm_at_least_90,
    unwrap_tensor_subclass,
)

try:
    import gemlite  # noqa: F401

    has_gemlite = True
except ModuleNotFoundError:
    has_gemlite = False


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


class XNNPackDynamicQuantizer(TwoStepQuantizer):
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


class TorchCompileDynamicQuantizer(Quantizer):
    def quantize(self, model: torch.nn.Module) -> torch.nn.Module:
        quantize_(model, int8_dynamic_activation_int8_weight())
        return model


def _ref_change_linear_weights_to_int8_dqtensors(model, filter_fn=None, **kwargs):
    """
    The deprecated implementation for int8 dynamic quant API, used as a reference for
    numerics and performance
    """
    from torchao.quantization.quant_api import (
        _get_subclass_inserter,
        _in_features_greater_than_16,
        _is_linear,
    )
    from torchao.quantization.subclass import Int8DynamicallyQuantizedLinearWeight

    if filter_fn is None:
        filter_fn = lambda *args: _is_linear(*args) and _in_features_greater_than_16(
            *args
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        _get_subclass_inserter(
            Int8DynamicallyQuantizedLinearWeight, enable_parametrization=False, **kwargs
        ),
        filter_fn,
    )


def _get_ref_change_linear_weights_to_woqtensors(deprecated_tenosr_subclass):
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
                deprecated_tenosr_subclass, enable_parametrization=True, **kwargs
            ),
            filter_fn,
        )

    return _ref_change_linear_weights_to_woqtensors


_ref_change_linear_weights_to_int8_woqtensors = (
    _get_ref_change_linear_weights_to_woqtensors(Int8WeightOnlyQuantizedLinearWeight)
)
_ref_change_linear_weights_to_int4_woqtensors = (
    _get_ref_change_linear_weights_to_woqtensors(Int4WeightOnlyQuantizedLinearWeight)
)


class TestQuantFlow(TestCase):
    GPU_DEVICES = (["cuda"] if torch.cuda.is_available() else []) + (
        ["xpu"] if torch.xpu.is_available() else []
    )

    def test_dynamic_quant_gpu_singleline(self):
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        quantize_(m, int8_dynamic_activation_int8_weight())
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

    @unittest.skipIf(not torch.xpu.is_available(), "Need XPU available")
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "only works for torch 2.8+")
    def test_int4_wo_quant_save_load(self):
        m = ToyLinearModel().eval().cpu()

        def api(model):
            quantize_(model, int4_weight_only(layout=Int4XPULayout()))
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
        m2 = m2.to(device="xpu")
        example_inputs = map(lambda x: x.xpu(), example_inputs)
        res = m2(*example_inputs)

        torch.testing.assert_close(ref, res.cpu())

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "only works for torch 2.4+")
    def test_int8_wo_quant_save_load(self):
        m = ToyLinearModel().eval().cpu()

        def api(model):
            quantize_(model, int8_weight_only())
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
        m2 = m2.to(device="cuda")
        example_inputs = map(lambda x: x.cuda(), example_inputs)
        res = m2(*example_inputs)

        # TODO: figure out why ROCm has a larger error
        atol, rtol = (1e-2, 1e-2) if torch.version.hip else (None, None)
        torch.testing.assert_close(ref, res.cpu(), atol=atol, rtol=rtol)

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, "skipping when torch verion is 2.3 or lower"
    )
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

    @unittest.skipIf(
        not TORCH_VERSION_AT_LEAST_2_3, "skipping when torch verion is 2.3 or lower"
    )
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

    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_quantizer_int4_weight_only(self):
        from torchao._models._eval import TransformerEvalWrapper
        from torchao.quantization.linear_quant_modules import Int4WeightOnlyQuantizer

        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path("../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = get_tokenizer(  # pyre-ignore[28]
            tokenizer_path,
            "Llama-2-7b-chat-hf",
        )
        groupsize = 64
        quantizer = Int4WeightOnlyQuantizer(
            groupsize,
        )
        model = quantizer.quantize(model).cuda()
        result = TransformerEvalWrapper(
            model,
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )
        assert result["results"]["wikitext"]["word_perplexity,none"] < 8.24, (
            f"accuracy regressed from 8.23 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_eval_wrapper(self):
        from torchao._models._eval import TransformerEvalWrapper

        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path("../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = get_tokenizer(  # pyre-ignore[28]
            tokenizer_path,
            "Llama-2-7b-chat-hf",
        )
        result = TransformerEvalWrapper(
            model,
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )
        assert result["results"]["wikitext"]["word_perplexity,none"] < 7.77, (
            f"accuracy regressed from 7.76 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    # EVAL IS CURRENTLY BROKEN FOR LLAMA 3, VERY LOW ACCURACY
    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_eval_wrapper_llama3(self):
        from torchao._models._eval import TransformerEvalWrapper

        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path(
            ".../gpt-fast/checkpoints/meta-llama/Meta-Llama-3-8B/model.pth"
        )
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = get_tokenizer(  # pyre-ignore[28]
            tokenizer_path,
            "Meta-Llama-3-8B",
        )
        result = TransformerEvalWrapper(
            model,
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )
        assert result["results"]["wikitext"]["word_perplexity,none"] < 8.24, (
            f"accuracy regressed from 8.23 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    # TODO: move to a separate test file
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @common_utils.parametrize(
        "mapping_type", [MappingType.SYMMETRIC, MappingType.SYMMETRIC_NO_CLIPPING_ERR]
    )
    def test_quantized_tensor_subclass_8da4w(self, mapping_type):
        group_size = 32
        m = ToyLinearModel().eval()
        m_copy = copy.deepcopy(m)
        example_inputs = m.example_inputs()
        quantize_(
            m,
            int8_dynamic_activation_int4_weight(
                group_size=group_size, mapping_type=mapping_type
            ),
        )

        assert isinstance(m.linear1.weight, LinearActivationQuantizedTensor)
        assert isinstance(m.linear2.weight, LinearActivationQuantizedTensor)
        assert isinstance(
            m.linear1.weight.original_weight_tensor, AffineQuantizedTensor
        )
        assert isinstance(
            m.linear2.weight.original_weight_tensor, AffineQuantizedTensor
        )

        # reference
        from torchao.quantization.linear_quant_modules import Int8DynActInt4WeightLinear
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

        quantizer = Int8DynActInt4WeightQuantizer(
            groupsize=group_size, mapping_type=mapping_type
        )
        m_copy = quantizer.quantize(m_copy)
        assert isinstance(m_copy.linear1, Int8DynActInt4WeightLinear)
        assert isinstance(m_copy.linear2, Int8DynActInt4WeightLinear)

        res = m(*example_inputs)
        ref = m_copy(*example_inputs)
        self.assertTrue(torch.equal(res, ref))

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    # @unittest.skipIf(TORCH_VERSION_AT_LEAST_2_5, "Test currently doesn't work for 2.5+")
    @unittest.skipIf(len(GPU_DEVICES) == 0, "Need GPU available")
    def test_quantized_tensor_subclass_int4(self):
        for device in self.GPU_DEVICES:
            # use 1024 so that we don't need padding
            m = ToyLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16).to(device)
            m_copy = copy.deepcopy(m)
            example_inputs = m.example_inputs(dtype=torch.bfloat16, device=device)

            group_size = 32
            if device == "xpu":
                quantize_(
                    m, int4_weight_only(group_size=group_size, layout=Int4XPULayout())
                )
            else:
                quantize_(m, int4_weight_only(group_size=group_size))
            assert isinstance(m.linear1.weight, AffineQuantizedTensor)
            assert isinstance(m.linear2.weight, AffineQuantizedTensor)

            # reference
            _ref_change_linear_weights_to_int4_woqtensors(m_copy, groupsize=group_size)

            res = m(*example_inputs)
            ref = m_copy(*example_inputs)

            self.assertTrue(torch.equal(res, ref))

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_int8_wo(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        m_copy = copy.deepcopy(m)
        example_inputs = tuple(map(lambda x: x.to(torch.bfloat16), m.example_inputs()))

        quantize_(m, int8_weight_only())

        assert isinstance(m.linear1.weight, AffineQuantizedTensor)
        assert isinstance(m.linear2.weight, AffineQuantizedTensor)

        # reference
        _ref_change_linear_weights_to_int8_woqtensors(m_copy)

        res = m(*example_inputs)
        ref = m_copy(*example_inputs)

        self.assertTrue(torch.equal(res, ref))

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(TORCH_VERSION_AT_LEAST_2_6, "Test only enabled for 2.5 and below")
    def test_quantized_tensor_subclass_int8_dyn_quant(self):
        # use multiples of 1024 so that we don't need padding
        m = ToyLinearModel(1024, 1024, 2048).eval().to(torch.bfloat16).to("cuda")
        m_copy = copy.deepcopy(m)
        # setting batch_size to 20 to be compatible with the kernel
        example_inputs = m.example_inputs(
            batch_size=20, dtype=torch.bfloat16, device="cuda"
        )
        quantize_(m, int8_dynamic_activation_int8_weight())

        assert isinstance(m.linear1.weight, LinearActivationQuantizedTensor)
        assert isinstance(m.linear2.weight, LinearActivationQuantizedTensor)
        assert isinstance(
            m.linear1.weight.original_weight_tensor, AffineQuantizedTensor
        )
        assert isinstance(
            m.linear2.weight.original_weight_tensor, AffineQuantizedTensor
        )

        # reference
        _ref_change_linear_weights_to_int8_dqtensors(m_copy)

        res = m(*example_inputs)
        ref = m_copy(*example_inputs)

        self.assertTrue(torch.equal(res, ref))

        # workaround for export path
        from torchao.utils import unwrap_tensor_subclass

        m_unwrapped = unwrap_tensor_subclass(m)

        m = torch.export.export(m_unwrapped, example_inputs, strict=True).module()
        exported_model_res = m(*example_inputs)

        self.assertTrue(torch.equal(exported_model_res, ref))

        # make sure it compiles
        torch._export.aot_compile(m_unwrapped, example_inputs)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_save_load(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        m_copy = copy.deepcopy(m)
        example_inputs = m.example_inputs(dtype=torch.bfloat16)

        quantize_(m, int8_weight_only())
        ref = m(*example_inputs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(m.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)

        m_copy.load_state_dict(state_dict, assign=True)

        res = m_copy(*example_inputs)
        self.assertEqual(res, ref)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_int8wo_quantized_model_to_device(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        example_inputs = m.example_inputs(dtype=torch.bfloat16, device="cpu")

        quantize_(m, int8_weight_only())
        ref = m(*example_inputs)

        example_inputs_cuda = (example_inputs[0].to("cuda"),)
        m.to(device="cuda")
        cuda_res = m(*example_inputs_cuda)
        self.assertEqual(cuda_res.cpu(), ref)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(TORCH_VERSION_AT_LEAST_2_5, "Test currently doesn't work for 2.5+")
    def test_int4wo_quantized_model_to_device(self):
        # TODO: change initial model to "cpu"
        devices = ["cuda", "cuda:0"]
        for device in devices:
            m = ToyLinearModel().eval().to(torch.bfloat16).to(device)
            example_inputs = m.example_inputs(dtype=torch.bfloat16, device=device)

            quantize_(m, int4_weight_only())
            ref = m(*example_inputs)

            example_inputs_cuda = (example_inputs[0].to(device),)
            m.to(device=device)
            cuda_res = m(*example_inputs_cuda)
            self.assertEqual(cuda_res.cpu(), ref)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_save_load_map_location(self):
        m = ToyLinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
        example_inputs = m.example_inputs(dtype=torch.bfloat16, device="cuda")

        quantize_(m, int8_weight_only())
        ref = m(*example_inputs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(m.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f.name, map_location="cpu", mmap=True)

        with torch.device("meta"):
            m_copy = ToyLinearModel().eval()

        m_copy.load_state_dict(state_dict, assign=True)
        m_copy.to(dtype=torch.bfloat16, device="cuda")

        res = m_copy(*example_inputs)
        self.assertEqual(res, ref)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_model_streaming(self):
        def reset_memory():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        reset_memory()
        m = ToyLinearModel()
        quantize_(m.to(device="cuda"), int8_weight_only())
        memory_baseline = torch.cuda.max_memory_allocated()

        del m
        reset_memory()
        m = ToyLinearModel()
        quantize_(m, int8_weight_only(), device="cuda")
        memory_streaming = torch.cuda.max_memory_allocated()

        for param in m.parameters():
            assert param.is_cuda
        self.assertLess(memory_streaming, memory_baseline)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("use_hqq", [True, False])
    def test_int4wo_cpu(self, dtype, x_dim, use_hqq):
        device = "cpu"
        m = ToyLinearModel().eval().to(dtype).to(device)
        example_inputs = m.example_inputs(dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)

        with torch.no_grad():
            quantize_(
                m,
                int4_weight_only(
                    group_size=32, layout=Int4CPULayout(), use_hqq=use_hqq
                ),
            )
            # ensure the expected op is in the code
            _, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            assert "_weight_int4pack_mm_for_cpu" in code[0]
            assert "aten.mm.default" not in code[0]

    # TODO(#1690): move to new config names
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @common_utils.parametrize(
        "config",
        [
            int4_weight_only(),
            float8_weight_only(),
            float8_dynamic_activation_float8_weight(),
            float8_static_activation_float8_weight(scale=torch.tensor([1.0])),
            int4_dynamic_activation_int4_weight(),
            int8_dynamic_activation_int8_weight(),
            int8_dynamic_activation_int4_weight(),
            int8_weight_only(),
            fpx_weight_only(ebits=4, mbits=3),
            gemlite_uintx_weight_only(),
            uintx_weight_only(dtype=torch.uint4),
        ],
    )
    @skip_if_rocm("ROCm enablement in progress")
    def test_workflow_e2e_numerics(self, config):
        """
        Simple test of e2e int4_weight_only workflow, comparing numerics
        to a bfloat16 baseline.
        """
        if (
            isinstance(
                config,
                (
                    float8_dynamic_activation_float8_weight,
                    float8_static_activation_float8_weight,
                ),
            )
            and not is_sm_at_least_89()
        ):
            return unittest.skip("requires CUDA capability 8.9 or greater")
        elif (
            isinstance(config, int4_dynamic_activation_int4_weight)
            and is_sm_at_least_90()
        ):
            return unittest.skip("only supported on CUDA capability 8.9, not greater")
        elif isinstance(config, gemlite_uintx_weight_only) and not has_gemlite:
            return unittest.skip("gemlite not available")

        # scale has to be moved to cuda here because the parametrization init
        # code happens before gating for cuda availability
        if isinstance(config, float8_static_activation_float8_weight):
            config.scale = config.scale.to("cuda")

        dtype = torch.bfloat16
        if isinstance(config, gemlite_uintx_weight_only):
            dtype = torch.float16

        # set up inputs
        x = torch.randn(128, 128, device="cuda", dtype=dtype)
        # TODO(future): model in float32 leads to error: https://gist.github.com/vkuzo/63b3bcd7818393021a6e3fb4ccf3c469
        # is that expected?
        m_ref = torch.nn.Sequential(torch.nn.Linear(128, 128)).cuda().to(dtype)
        m_q = copy.deepcopy(m_ref)

        # quantize
        quantize_(m_q, config)

        with torch.no_grad():
            y_ref = m_ref(x)
            y_q = m_q(x)

        sqnr = compute_error(y_ref, y_q)
        assert sqnr >= 16.5, f"SQNR {sqnr} is too low"

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_default(self):
        config1 = Int4WeightOnlyConfig(group_size=32)
        config2 = Int8WeightOnlyConfig()
        config = ModuleFqnToConfig({"_default": config1, "linear2": config2})
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        quantize_(model, config)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, AffineQuantizedTensor)
        assert isinstance(model.linear1.weight._layout, TensorCoreTiledLayout)
        assert isinstance(model.linear2.weight, AffineQuantizedTensor)
        assert isinstance(model.linear2.weight._layout, PlainLayout)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_module_name(self):
        config1 = Int4WeightOnlyConfig(group_size=32)
        config2 = Int8WeightOnlyConfig()
        config = ModuleFqnToConfig({"linear1": config1, "linear2": config2})
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        quantize_(model, config)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, AffineQuantizedTensor)
        assert isinstance(model.linear1.weight._layout, TensorCoreTiledLayout)
        assert isinstance(model.linear2.weight, AffineQuantizedTensor)
        assert isinstance(model.linear2.weight._layout, PlainLayout)

    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "Need torch 2.6+")
    def test_module_fqn_to_config_embedding_linear(self):
        weight_dtype = torch.int8
        granularity = PerGroup(8)
        mapping_type = MappingType.SYMMETRIC
        embedding_config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=mapping_type,
            scale_dtype=None,
        )
        # example model linear is Linear(16, 8)
        linear_config = Int8DynamicActivationInt4WeightConfig(group_size=16)

        config = ModuleFqnToConfig({"emb": embedding_config, "linear": linear_config})
        indices = torch.randint(0, 10, (32,))
        indices = indices.unsqueeze(0)
        example_inputs = (indices,)
        model = TestHelperModules.EmbeddingConvLinearModule().eval()
        model(*example_inputs)
        quantize_(
            model,
            config,
            filter_fn=lambda x, fqn: isinstance(x, torch.nn.Linear)
            or isinstance(x, torch.nn.Embedding),
        )
        model(*example_inputs)

        assert isinstance(model.emb.weight, AffineQuantizedTensor)
        assert isinstance(model.emb.weight._layout, QDQLayout)
        assert isinstance(model.linear.weight, LinearActivationQuantizedTensor)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_module_fqn_to_config_skip(self):
        config1 = Int4WeightOnlyConfig(group_size=32)
        config = ModuleFqnToConfig({"_default": config1, "linear2": None})
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        quantize_(model, config)
        model(*example_inputs)
        assert isinstance(model.linear1.weight, AffineQuantizedTensor)
        assert isinstance(model.linear1.weight._layout, TensorCoreTiledLayout)
        assert not isinstance(model.linear2.weight, AffineQuantizedTensor)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_int4wo_cuda_serialization(self):
        config = Int4WeightOnlyConfig(group_size=32)
        model = ToyLinearModel().cuda().to(dtype=torch.bfloat16)
        # quantize in cuda
        quantize_(model, config)
        example_inputs = model.example_inputs(device="cuda", dtype=torch.bfloat16)
        model(*example_inputs)
        with tempfile.NamedTemporaryFile() as ckpt:
            # save checkpoint in cuda
            torch.save(model.state_dict(), ckpt)
            # load checkpoint on cpu then move checkpoint to cuda
            # This is what torchtune does: https://github.com/pytorch/torchtune/blob/v0.6.1/torchtune/training/checkpointing/_utils.py#L253
            sd = torch.load(ckpt.name, weights_only=False, map_location="cpu")
            for k, v in sd.items():
                sd[k] = v.to("cuda")
            # load state_dict in cuda
            model.load_state_dict(sd, assign=True)


common_utils.instantiate_parametrized_tests(TestQuantFlow)


if __name__ == "__main__":
    unittest.main()
