# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
import unittest
import copy

import torch
import torch.nn as nn
from torchao.prototype.numerics import ObserverConfig, ObserverTensor
from torchao.quantization import quantize_, FqnToConfig
from torchao.testing.model_architectures import LlamaModelsLlama4Experts


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, k, bias=False)
        self.linear2 = torch.nn.Linear(k, n, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (
            torch.randn(
                batch_size, self.linear1.in_features, dtype=dtype, device=device
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestObserverTensor(unittest.TestCase):

    def test_toy_linear_passthrough(self):
        model = ToyLinearModel()
        model.eval().cuda()
        model_copy = copy.deepcopy(model)
        config = ObserverConfig()
        quantize_(model, config)

        inputs,  = model.example_inputs(batch_size=1, dtype=torch.float32, device="cuda")

        observed_outputs = model(inputs)
        expected_outputs = model_copy(inputs)   

        assert torch.allclose(observed_outputs, expected_outputs, atol=1e-2, rtol=1e-2)

    def test_llama4_passthrough(self):
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs

        m = LlamaModelsLlama4Experts(2, 1024, 1024, dtype=torch.bfloat16, device="cuda").eval()

        x = torch.randn(2, 1024,1024, 
            dtype=original_dtype,
        ).cuda()

        quant_config = FqnToConfig({
            'w1': ObserverConfig()
        })
        quantize_(m, quant_config, filter_fn=None)

        for i in range(1):
            print(f"calibrating {i}")
            m(x)
        
        breakpoint()
        
    def test_gptq_hf(self, device="cuda"):
        from transformers import AutoModelForCausalLM, TorchAoConfig, AutoTokenizer

        config = FqnToConfig(
            {
                r"model.layers.0.feed_forward.experts.down_proj": GPTQConfig()
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        model = AutoModelForCausalLM.from_pretrained(
            "jcaip/Llama-4-Scout-17B-two-layers-only-testing",
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quant_config,
        )

        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct")
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        asdf = model.model.layers[0].feed_forward.experts.down_proj
        breakpoint()
        result = asdf.gptq_quantize()

        print("content:", content)

    def test_gptq_with_input_recorder(self):
        from torchao.quantization.GPTQ import (
            Int4WeightOnlyGPTQQuantizer,
            MultiTensorInputRecorder,
        )
        from torchao._models.llama.model import (
            ModelArgs,
            Transformer,
            prepare_inputs_for_model,
        )
        from torchao.quantization import Int4WeightOnlyConfig

        torch.set_default_dtype(torch.bfloat16)

        config = ModelArgs(n_layer=2)

        with torch.device("cuda"):
            model = Transformer(config)
            model.setup_caches(max_batch_size=2, max_seq_length=100)
            idx = torch.randint(1, 10000, (10, 2, 50)).to(torch.int32)
            test_input = prepare_inputs_for_model(idx[0])
        

        # get new gptq implementation out
        gptqnew_config = ObserverConfig()
        quantize_(model, gptqnew_config)

        # new calibration
        for i in range(10):
            input = prepare_inputs_for_model(idx[i])
            model(*input)

        convert_config = ObserverConfig(step="convert")

        quantize_(model, convert_config)

        breakpoint()


        # get old gptq implementation out
        model2 = copy.deepcopy(model)
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

        from torchao.quantization.utils import compute_error
        self.assertGreater(compute_error(outgptq, out), 30)
        self.assertGreater(compute_error(outgptq, out), compute_error(outq, out))
        torch.set_default_dtype(torch.float32)
        print(outgptq)
        print(out)
        breakpoint()


    def test_asdf(self):
        from torchao.quantization.GPTQ import (
            Int4WeightOnlyGPTQQuantizer,
            MultiTensorInputRecorder,
        )
        from torchao._models.llama.model import (
            ModelArgs,
            Transformer,
            prepare_inputs_for_model,
        )
        from torchao.quantization import Int4WeightOnlyConfig

        torch.set_default_dtype(torch.bfloat16)

        config = ModelArgs(n_layer=2)

        with torch.device("cuda"):
            model = Transformer(config)
            model.setup_caches(max_batch_size=2, max_seq_length=100)
            idx = torch.randint(1, 10000, (10, 2, 50)).to(torch.int32)
            test_input = prepare_inputs_for_model(idx[0])
        
        model2 = copy.deepcopy(model)
        out = model(*test_input)
        quantize_(model2, Int4WeightOnlyConfig(version=2))

        breakpoint()

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

        from torchao.quantization.utils import compute_error
        self.assertGreater(compute_error(outgptq, out), 30)
        self.assertGreater(compute_error(outgptq, out), compute_error(outq, out))
        torch.set_default_dtype(torch.float32)
        print(outgptq)
        print(out)
        breakpoint()
          

if __name__ == "__main__":
    unittest.main()
