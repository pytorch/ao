# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest
import torch
import os
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

import torchao
from torchao.dtypes import (
    AffineQuantizedTensor,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.subclass import (
    LinearActQuantizedTensor,
    Int8WeightOnlyQuantizedLinearWeight,
    Int4WeightOnlyQuantizedLinearWeight,
)
from torchao import quantize
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    Quantizer,
    TwoStepQuantizer,
    int8_dynamic_activation_int4_weight,
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
)
from torchao.utils import (
    TORCH_VERSION_AFTER_2_3,
    TORCH_VERSION_AFTER_2_4,
)
from pathlib import Path
from torchao._models.llama.tokenizer import get_tokenizer
from torchao._models.llama.model import Transformer, prepare_inputs_for_model
import copy
import tempfile
from torch.testing._internal.common_utils import TestCase


def dynamic_quant(model, example_inputs):
    m = torch.export.export(model, example_inputs).module()
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))
    m = prepare_pt2e(m, quantizer)
    m = convert_pt2e(m)
    return m

def capture_and_prepare(model, example_inputs):
    m = torch.export.export(model, example_inputs)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))
    m = prepare_pt2e(m, quantizer)
    # TODO: we can run the weight observer in convert_pt2e so that user don't need to run this
    m(*example_inputs)
    return m

class XNNPackDynamicQuantizer(TwoStepQuantizer):

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_with_custom_fn_if_matches_filter(
            model,
            lambda linear_mod: capture_and_prepare(linear_mod, (torch.randn(1, linear_mod.in_features))),
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
        quantize(model, int8_dynamic_activation_int8_weight())
        return model

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False).to(torch.float)
        self.linear2 = torch.nn.Linear(n, k, bias=False).to(torch.float)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def _ref_change_linear_weights_to_int8_dqtensors(model, filter_fn=None, **kwargs):
    """
    The deprecated implementation for int8 dynamic quant API, used as a reference for
    numerics and performance
    """
    from torchao.quantization.quant_api import _in_features_greater_than_16
    from torchao.quantization.quant_api import _is_linear
    from torchao.quantization.quant_api import _get_subclass_inserter
    from torchao.quantization.subclass import Int8DynamicallyQuantizedLinearWeight

    if filter_fn is None:
        filter_fn = lambda *args: _is_linear(*args) and _in_features_greater_than_16(
            *args
        )

    _replace_with_custom_fn_if_matches_filter(
        model, _get_subclass_inserter(Int8DynamicallyQuantizedLinearWeight, enable_parametrization=False, **kwargs), filter_fn
    )

def _get_ref_change_linear_weights_to_woqtensors(deprecated_tenosr_subclass):
    def _ref_change_linear_weights_to_woqtensors(model, filter_fn=None, **kwargs):
        """
        The deprecated implementation for weight only quant API, used as a reference for
        numerics and performance
        """
        from torchao.quantization.quant_api import _is_linear
        from torchao.quantization.quant_api import _get_subclass_inserter

        filter_fn = kwargs.pop("filter_fn", _is_linear)

        _replace_with_custom_fn_if_matches_filter(
            model,
            _get_subclass_inserter(deprecated_tenosr_subclass, enable_parametrization=True, **kwargs),
            filter_fn,
        )

    return _ref_change_linear_weights_to_woqtensors

_ref_change_linear_weights_to_int8_woqtensors = _get_ref_change_linear_weights_to_woqtensors(Int8WeightOnlyQuantizedLinearWeight)
_ref_change_linear_weights_to_int4_woqtensors = _get_ref_change_linear_weights_to_woqtensors(Int4WeightOnlyQuantizedLinearWeight)

class TestQuantFlow(TestCase):
    def test_dynamic_quant_gpu_singleline(self):
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        m = quantize(m, int8_dynamic_activation_int8_weight())
        quantized = m(*example_inputs)
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

    @unittest.skip("FAILED test/quantization/test_quant_api.py::TestQuantFlow::test_dynamic_quant_gpu_unified_api_eager_mode_impl - AssertionError: Tensor-likes are not equal!")
    def test_dynamic_quant_gpu_unified_api_eager_mode_impl(self):
        quantizer = TorchCompileDynamicQuantizer()
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        quantized = m(*example_inputs)
        m = torch.compile(m, mode="max-autotune")
        compiled = m(*example_inputs)
        torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "only works for torch 2.4+")
    def test_int8_wo_quant_save_load(self):
        from torchao.quantization.quant_api import (
            change_linear_weights_to_int8_woqtensors,
        )
        m = ToyLinearModel().eval().cpu()
        def api(model):
            model = quantize(model, int8_weight_only())

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

        torch.testing.assert_close(ref, res.cpu())

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_3, "skipping when torch verion is 2.3 or lower")
    def test_8da4w_quantizer(self):
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer
        from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear

        quantizer = Int8DynActInt4WeightQuantizer(groupsize=32)
        m = ToyLinearModel().eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        assert isinstance(m.linear1, Int8DynActInt4WeightLinear)
        assert isinstance(m.linear2, Int8DynActInt4WeightLinear)
        m(*example_inputs)

    # TODO: save model weights as artifacts and re-enable in CI
    # For now, to run this test, you will need to download the weights from HF
    # and run this script to convert them:
    # https://github.com/pytorch-labs/gpt-fast/blob/6253c6bb054e658d67566150f87329b87815ae63/scripts/convert_hf_checkpoint.py
    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_8da4w_gptq_quantizer(self):
        from torchao.quantization.GPTQ import Int8DynActInt4WeightGPTQQuantizer
        from torchao._models._eval import InputRecorder, TransformerEvalWrapper
        # should be similar to TorchCompileDynamicQuantizer
        precision = torch.bfloat16
        device = "cpu"
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
        blocksize = 128
        percdamp = 0.01
        groupsize = 128
        calibration_tasks = ["wikitext"]
        calibration_limit = 1
        calibration_seq_length = 100
        input_prep_func = prepare_inputs_for_model
        pad_calibration_inputs = False

        inputs = InputRecorder(
            tokenizer,
            calibration_seq_length,
            input_prep_func,
            pad_calibration_inputs,
            model.config.vocab_size,
        ).record_inputs(
            calibration_tasks,
            calibration_limit,
        ).get_inputs()

        quantizer = Int8DynActInt4WeightGPTQQuantizer(
            blocksize,
            percdamp,
            groupsize,
            precision=precision,
        )
        model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)
        model = quantizer.quantize(model, inputs)
        result=TransformerEvalWrapper(
            model,
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )

        assert result['results']['wikitext']['word_perplexity,none'] < 7.88, (
            f"accuracy regressed from 7.87 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "skipping when torch verion is 2.4 or lower")
    def test_8da4w_quantizer_eval(self):
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer
        from torchao._models._eval import TransformerEvalWrapper

        precision = torch.bfloat16
        device = "cpu"
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

        quantizer = Int8DynActInt4WeightQuantizer(groupsize=128, precision=precision)
        q_model = quantizer.quantize(model)
        result=TransformerEvalWrapper(
            q_model,
            tokenizer,
            q_model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )
        assert result['results']['wikitext']['word_perplexity,none'] < 8.24, (
            f"accuracy regressed from 8.23 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_gptq_quantizer_int4_weight_only(self):
        from torchao.quantization.GPTQ import Int4WeightOnlyGPTQQuantizer
        from torchao._models._eval import InputRecorder, TransformerEvalWrapper
        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path("../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device="cpu")
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = get_tokenizer(  # pyre-ignore[28]
            tokenizer_path,
            "Llama-2-7b-chat-hf",
        )
        blocksize = 128
        percdamp = 0.01
        groupsize = 64
        calibration_tasks = ["wikitext"]
        calibration_limit = 1
        calibration_seq_length = 100
        input_prep_func = prepare_inputs_for_model
        pad_calibration_inputs = False

        inputs = InputRecorder(
            tokenizer,
            calibration_seq_length,
            input_prep_func,
            pad_calibration_inputs,
            model.config.vocab_size,
            device="cpu",
        ).record_inputs(
            calibration_tasks,
            calibration_limit,
        ).get_inputs()

        quantizer = Int4WeightOnlyGPTQQuantizer(
            blocksize,
            percdamp,
            groupsize,
        )
        model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)

        model = quantizer.quantize(model, inputs).cuda()
        result = TransformerEvalWrapper(
            model.cuda(),
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )
        assert result['results']['wikitext']['word_perplexity,none'] < 7.77, (
            f"accuracy regressed from 7.76 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_quantizer_int4_weight_only(self):
        from torchao.quantization.GPTQ import Int4WeightOnlyQuantizer
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
        assert result['results']['wikitext']['word_perplexity,none'] < 8.24, (
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
        result=TransformerEvalWrapper(
            model,
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            1,
        )
        assert result['results']['wikitext']['word_perplexity,none']<7.77, (
            f"accuracy regressed from 7.76 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    # EVAL IS CURRENTLY BROKEN FOR LLAMA 3, VERY LOW ACCURACY
    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_eval_wrapper_llama3(self):
        from torchao._models._eval import TransformerEvalWrapper
        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path(".../gpt-fast/checkpoints/meta-llama/Meta-Llama-3-8B/model.pth")
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
        assert result['results']['wikitext']['word_perplexity,none'] < 8.24, (
            f"accuracy regressed from 8.23 to {result['results']['wikitext']['word_perplexity,none']}"
        )

    # TODO: move to a separate test file
    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "Test only enabled for 2.4+")
    def test_quantized_tensor_subclass_8da4w(self):
        group_size = 32
        m = ToyLinearModel().eval()
        m_copy = copy.deepcopy(m)
        example_inputs = m.example_inputs()
        m = quantize(m, int8_dynamic_activation_int4_weight(group_size=group_size))

        assert isinstance(m.linear1.weight, LinearActQuantizedTensor)
        assert isinstance(m.linear2.weight, LinearActQuantizedTensor)
        assert isinstance(m.linear1.weight.original_weight_tensor, AffineQuantizedTensor)
        assert isinstance(m.linear2.weight.original_weight_tensor, AffineQuantizedTensor)

        # reference
        from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer
        from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear

        quantizer = Int8DynActInt4WeightQuantizer(groupsize=group_size)
        m_copy = quantizer.quantize(m_copy)
        assert isinstance(m_copy.linear1, Int8DynActInt4WeightLinear)
        assert isinstance(m_copy.linear2, Int8DynActInt4WeightLinear)

        res = m(*example_inputs)
        ref = m_copy(*example_inputs)
        self.assertTrue(torch.equal(res, ref))

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_int4(self):
        # use 1024 so that we don't need padding
        m = ToyLinearModel(1024, 1024, 1024).eval().to(torch.bfloat16).to("cuda")
        m_copy = copy.deepcopy(m)
        example_inputs = m.example_inputs(dtype=torch.bfloat16, device="cuda")

        group_size = 32
        m = quantize(m, int4_weight_only(group_size=group_size))
        assert isinstance(m.linear1.weight, AffineQuantizedTensor)
        assert isinstance(m.linear2.weight, AffineQuantizedTensor)

        # reference
        _ref_change_linear_weights_to_int4_woqtensors(m_copy, groupsize=group_size)

        res = m(*example_inputs)
        ref = m_copy(*example_inputs)

        self.assertTrue(torch.equal(res, ref))


    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_int8_wo(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        m_copy = copy.deepcopy(m)
        example_inputs = tuple(map(lambda x: x.to(torch.bfloat16), m.example_inputs()))

        m = quantize(m, int8_weight_only())

        assert isinstance(m.linear1.weight, AffineQuantizedTensor)
        assert isinstance(m.linear2.weight, AffineQuantizedTensor)

        # reference
        _ref_change_linear_weights_to_int8_woqtensors(m_copy)


        res = m(*example_inputs)
        ref = m_copy(*example_inputs)

        self.assertTrue(torch.equal(res, ref))


    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_int8_dyn_quant(self):
        # use multiples of 1024 so that we don't need padding
        m = ToyLinearModel(1024, 1024, 2048).eval().to(torch.bfloat16).to("cuda")
        m_copy = copy.deepcopy(m)
        # setting batch_size to 20 to be compatible with the kernel
        example_inputs = m.example_inputs(batch_size=20, dtype=torch.bfloat16, device="cuda")
        m = quantize(m, int8_dynamic_activation_int8_weight())

        assert isinstance(m.linear1.weight, LinearActQuantizedTensor)
        assert isinstance(m.linear2.weight, LinearActQuantizedTensor)
        assert isinstance(m.linear1.weight.original_weight_tensor, AffineQuantizedTensor)
        assert isinstance(m.linear2.weight.original_weight_tensor, AffineQuantizedTensor)

        # reference
        _ref_change_linear_weights_to_int8_dqtensors(m_copy)

        res = m(*example_inputs)
        ref = m_copy(*example_inputs)

        self.assertTrue(torch.equal(res, ref))

        m_copy = copy.deepcopy(m)
        exported_m = torch.export.export(m, example_inputs).module()
        exported_model_res = exported_m(*example_inputs)

        self.assertTrue(torch.equal(exported_model_res, ref))

        # make sure it compiles
        torch._export.aot_compile(m_copy, example_inputs)

    @unittest.skipIf(not TORCH_VERSION_AFTER_2_4, "Test only enabled for 2.4+")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_quantized_tensor_subclass_save_load(self):
        m = ToyLinearModel().eval().to(torch.bfloat16)
        m_copy = copy.deepcopy(m)
        example_inputs = m.example_inputs(dtype=torch.bfloat16)

        m = quantize(m, int8_weight_only())
        ref = m(*example_inputs)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(m.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)

        m_copy.load_state_dict(state_dict, assign=True)

        res = m_copy(*example_inputs)
        self.assertEqual(res, ref)


if __name__ == "__main__":
    unittest.main()
