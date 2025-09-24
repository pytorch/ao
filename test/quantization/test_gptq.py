# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_utils import TestCase

from torchao._models.llama.model import (
    ModelArgs,
    Transformer,
    prepare_inputs_for_model,
)
from torchao._models.llama.tokenizer import get_tokenizer
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.utils import compute_error

torch.manual_seed(0)


class TestGPTQ(TestCase):
    @unittest.skip("skipping until we get checkpoints for gpt-fast")
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_gptq_quantizer_int4_weight_only(self):
        from torchao._models._eval import (
            LMEvalInputRecorder,
            TransformerEvalWrapper,
        )
        from torchao.quantization.GPTQ import Int4WeightOnlyGPTQQuantizer

        precision = torch.bfloat16
        device = "cuda"
        checkpoint_path = Path(
            "../../checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
        )
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
        groupsize = 64
        blocksize = 128
        percdamp = 0.01
        calibration_tasks = ["wikitext"]
        calibration_limit = 1
        calibration_seq_length = 100
        input_prep_func = prepare_inputs_for_model
        pad_calibration_inputs = False
        inputs = (
            LMEvalInputRecorder(
                tokenizer,
                calibration_seq_length,
                input_prep_func,
                model.config.vocab_size,
                pad_calibration_inputs,
                device="cpu",
            )
            .record_inputs(
                calibration_tasks,
                calibration_limit,
            )
            .get_recorded_inputs()
        )

        quantizer = Int4WeightOnlyGPTQQuantizer(
            groupsize,
            blocksize,
            percdamp,
        )
        model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)

        model = quantizer.quantize(model, *inputs).cuda()

        model.reset_caches()
        with torch.device("cuda"):
            model.setup_caches(max_batch_size=1, max_seq_length=model.config.block_size)

        limit = 1
        result = TransformerEvalWrapper(
            model.cuda(),
            tokenizer,
            model.config.block_size,
            prepare_inputs_for_model,
            device,
        ).run_eval(
            ["wikitext"],
            limit,
        )

        assert result["results"]["wikitext"]["word_perplexity,none"] < 7.77, (
            f"accuracy regressed from 7.76 to {result['results']['wikitext']['word_perplexity,none']}"
        )


class TestMultiTensorFlow(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_multitensor_add_tensors(self):
        from torchao.quantization.GPTQ import MultiTensor

        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)
        mt = MultiTensor(tensor1)
        mt.add_tensors(tensor2)
        self.assertEqual(mt.count, 2)
        self.assertTrue(torch.equal(mt.values[0], tensor1))
        self.assertTrue(torch.equal(mt.values[1], tensor2))

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_multitensor_pad_unpad(self):
        from torchao.quantization.GPTQ import MultiTensor

        tensor1 = torch.randn(3, 3)
        mt = MultiTensor(tensor1)
        mt.pad_to_length(3)
        self.assertEqual(mt.count, 3)
        mt.unpad()
        self.assertEqual(mt.count, 1)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_multitensor_inplace_operation(self):
        from torchao.quantization.GPTQ import MultiTensor

        tensor1 = torch.ones(3, 3)
        mt = MultiTensor(tensor1)
        mt += 1  # In-place addition
        self.assertTrue(torch.equal(mt.values[0], torch.full((3, 3), 2)))


class TestMultiTensorInputRecorder(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_multitensor_input_recorder(self):
        from torchao.quantization.GPTQ import MultiTensor, MultiTensorInputRecorder

        input_recorder = MultiTensorInputRecorder()
        in1 = ([1], torch.randn(3, 3), (1, "dog", torch.randn(3, 3)), torch.float)
        in2 = ([1], torch.randn(3, 3), (1, "dog", torch.randn(3, 3)), torch.float)

        input_recorder(*in1)
        input_recorder(*in2)

        MT_input = input_recorder.get_recorded_inputs()

        self.assertEqual(MT_input[0], [1])
        self.assertTrue(isinstance(MT_input[1], MultiTensor))
        self.assertTrue(isinstance(MT_input[2], tuple))
        self.assertEqual(MT_input[2][0], 1)
        self.assertEqual(MT_input[2][1], "dog")
        self.assertTrue(isinstance(MT_input[2][2], MultiTensor))
        self.assertEqual(MT_input[3], torch.float)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_gptq_with_input_recorder(self):
        from torchao.quantization.GPTQ import (
            Int4WeightOnlyGPTQQuantizer,
            MultiTensorInputRecorder,
        )

        torch.set_default_dtype(torch.bfloat16)

        config = ModelArgs(n_layer=2)

        with torch.device("cuda"):
            model = Transformer(config)
            model.setup_caches(max_batch_size=2, max_seq_length=100)
            idx = torch.randint(1, 10000, (10, 2, 50)).to(torch.int32)
            test_input = prepare_inputs_for_model(idx[0])
        import copy

        model2 = copy.deepcopy(model)
        out = model(*test_input)
        quantize_(model2, Int4WeightOnlyConfig(version=1))

        outq = model2(*test_input)
        del model2

        input_recorder = MultiTensorInputRecorder()
        for i in range(10):
            input = prepare_inputs_for_model(idx[i])
            input_recorder(*input)

        args = input_recorder.get_recorded_inputs()

        quantizer = Int4WeightOnlyGPTQQuantizer()

        quantizer.quantize(model, *args)

        outgptq = model(*test_input)

        self.assertGreater(compute_error(outgptq, out), 30)
        self.assertGreater(compute_error(outgptq, out), compute_error(outq, out))
        torch.set_default_dtype(torch.float32)


if __name__ == "__main__":
    unittest.main()
