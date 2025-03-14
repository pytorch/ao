import unittest
from typing import Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from torchao.prototype.quantization.module_swap.data_getters import LLMPTQDataGetter

test_config = LlamaConfig(
    vocab_size=10,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
)


def get_test_llama_model_data() -> Tuple[LlamaForCausalLM, torch.Tensor]:
    model = LlamaForCausalLM(test_config)
    input_ids = torch.randint(0, test_config.vocab_size, (1, 10))
    return model, input_ids


class TestPTQDataGetter(unittest.TestCase):
    @unittest.skip("TypeError: cannot unpack non-iterable NoneType object")
    def test_data_getter(self) -> None:
        model, data = get_test_llama_model_data()
        data_getter = LLMPTQDataGetter(model, data, 1)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                data = data_getter.pop(model, name)


if __name__ == "__main__":
    unittest.main()
