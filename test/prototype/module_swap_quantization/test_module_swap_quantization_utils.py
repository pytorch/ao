import unittest

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from torchao.prototype.quantization.module_swap import QuantizedLinear
from torchao.prototype.quantization.module_swap.module_swap import (
    QuantizationRecipe,
    replace_all_linear_with_quantized_linear,
)
from torchao.prototype.quantization.module_swap.utils import set_bit_widths_by_name

test_config = LlamaConfig(
    vocab_size=10,
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=64,
)

base_recipe = QuantizationRecipe(
    weight_bits=4,
    weight_group_size=32,
    weight_quantization=True,
    dynamic_weights=False,
    activation_bits=8,
    activation_group_size="per_token",
    activation_quantization=True,
    input_quantization=True,
    output_quantization=True,
    dynamic_activations=True,
    range_learning=False,
    exclude_layers=["lm_head"],
)


def get_test_llama_model_data() -> tuple[LlamaForCausalLM, torch.Tensor]:
    model = LlamaForCausalLM(test_config)
    input_ids = torch.randint(0, test_config.vocab_size, (1, 10))
    return model, input_ids


class TestQuantizedModuleUtils(unittest.TestCase):
    def test_set_bit_widths_by_name(self) -> None:
        model, _ = get_test_llama_model_data()
        replace_all_linear_with_quantized_linear(model, base_recipe)

        bit_width_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                bit_width_dict[name] = {"weight": 7, "activation": 9}

        set_bit_widths_by_name(model, bit_width_dict)

        for _, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                assert module.weight_quantizer.num_bits == 7
                assert module.input_quantizer is not None
                assert module.input_quantizer.num_bits == 9
                assert module.output_quantizer is not None
                assert module.output_quantizer.num_bits == 9


if __name__ == "__main__":
    unittest.main()
