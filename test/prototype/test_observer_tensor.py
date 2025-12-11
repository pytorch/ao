# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch

from torchao.prototype.numerics import (
    GPTQConfig,
    ObserverConfig,
    sequential_quantize_,
)
from torchao.quantization import FqnToConfig, Int4WeightOnlyConfig, quantize_
from torchao.quantization.utils import compute_error
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
    def test_sequential_linear(self):
        model = ToyLinearModel()
        model.to(dtype=torch.bfloat16).eval().cuda()
        model_copy = copy.deepcopy(model)

        (inputs,) = model.example_inputs(
            batch_size=1, dtype=torch.bfloat16, device="cuda"
        )

        out_baseline = model_copy(inputs)

        config = ObserverConfig()
        quantize_(model, config)

        quantize_(model_copy, ObserverConfig("observe_all"))

        for i in range(1):
            model(inputs)
            model_copy(inputs)

        sequential_quantize_(model, GPTQConfig(), inputs)

        out_sequential_quant = model(inputs)

        quantize_(model_copy, GPTQConfig())
        out_nonsequential_quant = model_copy(inputs)

        print(
            "sequential quant error: ",
            compute_error(out_sequential_quant, out_baseline),
        )
        print(
            "non-sequential quant error: ",
            compute_error(out_nonsequential_quant, out_baseline),
        )
        breakpoint()

    def test_toy_linear_passthrough(self):
        model = ToyLinearModel()
        model.eval().cuda()
        model_copy = copy.deepcopy(model)
        config = ObserverConfig()
        quantize_(model, config)

        (inputs,) = model.example_inputs(
            batch_size=1, dtype=torch.float32, device="cuda"
        )

        observed_outputs = model(inputs)
        expected_outputs = model_copy(inputs)

        assert torch.allclose(observed_outputs, expected_outputs, atol=1e-2, rtol=1e-2)

    def test_llama4_passthrough(self):
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs

        m = LlamaModelsLlama4Experts(
            2, 1024, 512, dtype=torch.bfloat16, device="cuda"
        ).eval()

        x = torch.randn(
            2,
            1024,
            1024,
            dtype=original_dtype,
        ).cuda()

        quant_config = FqnToConfig({"w1": ObserverConfig()})
        quantize_(m, quant_config, filter_fn=None)

        for i in range(1):
            print(f"calibrating {i}")
            m(x)

    def test_gptq_hf(self, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

        config = FqnToConfig(
            {
                r"model.layers.0.feed_forward.experts.down_proj": ObserverConfig()
                # r"model.layers.0.feed_forward.experts.gate_up_proj": Int4WeightOnlyConfig()
            }
        )
        quant_config = TorchAoConfig(quant_type=config)
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct",
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quant_config,
        )
        # model.model.layers[0].feed_forward.experts.gate_up_proj = nn.Parameter(
        #     model.model.layers[0].feed_forward.experts.gate_up_proj.transpose(-2, -1).contiguous()
        # )
        quantize_(model, config, filter_fn=None)

        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Llama-4-Scout-17B-16E-Instruct"
        )
        prompt = "Give me a short introduction to large language model."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("content:", content)

        breakpoint()

        convert_config = FqnToConfig(
            {
                r"model.layers.0.feed_forward.experts.down_proj": ObserverConfig(
                    step="convert"
                )
            }
        )
        quantize_(model, convert_config, filter_fn=None)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("content:", content)

    def test_gptq_with_input_recorder(self):
        torch.manual_seed(43)
        from torchao._models.llama.model import (
            ModelArgs,
            Transformer,
            prepare_inputs_for_model,
        )

        torch.set_default_dtype(torch.bfloat16)

        config = ModelArgs(n_layer=2)

        with torch.device("cuda"):
            model = Transformer(config)
            model.setup_caches(max_batch_size=2, max_seq_length=100)
            idx = torch.randint(1, 10000, (10, 2, 50)).to(torch.int32)
            test_input = prepare_inputs_for_model(idx[0])

            model2 = copy.deepcopy(model)
            model_baseline = copy.deepcopy(model)

            # get new gptq implementation out
            gptqnew_config = ObserverConfig()
            quantize_(model, gptqnew_config)

            # new calibration
            for i in range(10):
                input = prepare_inputs_for_model(idx[i])
                model(*input)

            convert_config = GPTQConfig()
            quantize_(model, convert_config)
            out_gptq = model(*test_input)

            quantize_(model2, Int4WeightOnlyConfig(version=2))
            out_rtn = model2(*test_input)

            out = model_baseline(*test_input)

            from torchao.quantization.utils import compute_error

            print("rtn: ", compute_error(out_rtn, out))
            print("new gptq: ", compute_error(out_gptq, out))


if __name__ == "__main__":
    unittest.main()
