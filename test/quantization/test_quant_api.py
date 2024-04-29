# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
# This test takes a long time to run
import unittest
import torch
import os
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    apply_dynamic_quant,
    apply_weight_only_int8_quant,
    Quantizer,
    TwoStepQuantizer,
)
from torchao.quantization.utils import (
    TORCH_VERSION_AFTER_2_3,
    TORCH_VERSION_AFTER_2_4,
)
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model import Transformer, prepare_inputs_for_model


def dynamic_quant(model, example_inputs):
    m = capture_pre_autograd_graph(model, example_inputs)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))
    m = prepare_pt2e(m, quantizer)
    m = convert_pt2e(m)
    return m

def _apply_dynamic_quant(model):
    """
    Applies dynamic symmetric per-token activation and per-channel weight
    quantization to all linear layers in the given model using
    module swaps.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda linear_mod: dynamic_quant(linear_mod, (torch.randn(1, linear_mod.in_features))),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )
    return model


def capture_and_prepare(model, example_inputs):
    m = capture_pre_autograd_graph(model, example_inputs)
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
        apply_dynamic_quant(model)
        return model

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32, bias=False).to(torch.float)
        self.linear2 = torch.nn.Linear(32, 64, bias=False).to(torch.float)

    def example_inputs(self):
        return (torch.randn(1, 64).to(torch.float),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class TestQuantFlow(unittest.TestCase):
    def test_dynamic_quant_gpu_singleline(self):
        m = M().eval()
        m = _apply_dynamic_quant(m)
        quantized = m(*m.example_inputs())
        # AssertionError: Expecting input to have dtype torch.float32, but got dtype: torch.float64
        # While executing %choose_qparams_tensor_1 : [num_users=2] = call_function[target=torch.ops.quantized_decomposed.choose_qparams.tensor](args = (%arg0_3, -128, 127, 0.000244140625, torch.int8), kwargs = {})
        # m = torch.compile(m, mode="max-autotune")
        # print(example_inputs[0].dtype)
        # compiled = m(*example_inputs)
        # torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    @unittest.skip("skipping for now due to torch.compile error")
    def test_dynamic_quant_gpu_unified_api_unified_impl(self):
        quantizer = XNNPackDynamicQuantizer()
        m = M().eval()
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
        m = M().eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        quantized = m(*example_inputs)
        m = torch.compile(m, mode="max-autotune")
        compiled = m(*example_inputs)
        torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_int8_wo_quant_save_load(self):
        m = M().eval().cpu()
        apply_weight_only_int8_quant(m)
        example_inputs = m.example_inputs()
        ref = m(*example_inputs)
        _TMP_FN = "_test.pt"
        torch.save(m.state_dict(), _TMP_FN)

        state_dict = torch.load(_TMP_FN)
        os.remove(_TMP_FN)
        m2 = M().eval()
        apply_weight_only_int8_quant(m2)
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
        m = M().eval()
        example_inputs = m.example_inputs()
        m = quantizer.quantize(m)
        assert isinstance(m.linear1, Int8DynActInt4WeightLinear)
        assert isinstance(m.linear2, Int8DynActInt4WeightLinear)
        m(*example_inputs)

    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    def test_8da4w_gptq_quantizer(self):
        from torchao.quantization.GPTQ import Int8DynActInt4WeightGPTQQuantizer, InputRecorder, TransformerEvalWrapper
        # should be similar to TorchCompileDynamicQuantizer
        precision = torch.bfloat16
        device = "cpu"
        checkpoint_path = Path("../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
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
        from torchao.quantization.GPTQ import TransformerEvalWrapper

        precision = torch.bfloat16
        device = "cpu"
        checkpoint_path = Path("../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
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
    def test_gptq_quantizer_int4wo(self):
        from torchao.quantization.GPTQ import Int4WeightOnlyGPTQQuantizer, InputRecorder, TransformerEvalWrapper
        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path("../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device="cpu")
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
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
    def test_quantizer_int4wo(self):
        from torchao.quantization.GPTQ import Int4WeightOnlyQuantizer, TransformerEvalWrapper
        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path("../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
        )
        groupsize = 128
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
        from torchao.quantization.GPTQ import TransformerEvalWrapper
        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path("../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")
        model = Transformer.from_name(checkpoint_path.parent.name)
        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(dtype=precision, device=device)
        model.eval()
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(  # pyre-ignore[28]
            model_file=str(tokenizer_path)
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

if __name__ == "__main__":
    unittest.main()
