
import unittest
import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import quantize_
from torchao.quantization.transform_module import register_quantize_module_handler


class TestModuleSwapConfig(AOBaseConfig):
    pass


@register_quantize_module_handler(TestModuleSwapConfig)
def _test_module_swap_transform(module, config):
    # Create a new module that replaces the original one
    new_module = nn.Linear(module.in_features, module.out_features)
    new_module.weight = torch.nn.Parameter(torch.ones_like(module.weight))
    new_module.bias = torch.nn.Parameter(torch.ones_like(module.bias) if module.bias is not None else None)
    return new_module


class TestRegisterQuantizeModuleHandler(unittest.TestCase):
    def test_top_level_module_swap(self):
        # Test that module swapping works for top-level modules
        model = nn.Linear(10, 10)
        original_weight = model.weight.clone()
        
        # Apply quantization that swaps the module
        quantize_(model, TestModuleSwapConfig())
        
        # Check that the module was actually swapped
        self.assertFalse(torch.equal(model.weight, original_weight))
        self.assertTrue(torch.allclose(model.weight, torch.ones_like(model.weight)))
        
    def test_nested_module_swap(self):
        # Test that module swapping works for nested modules
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        original_weight1 = model[0].weight.clone()
        original_weight2 = model[2].weight.clone()
        
        # Apply quantization that swaps the modules
        quantize_(model, TestModuleSwapConfig())
        
        # Check that the modules were actually swapped
        self.assertFalse(torch.equal(model[0].weight, original_weight1))
        self.assertFalse(torch.equal(model[2].weight, original_weight2))
        self.assertTrue(torch.allclose(model[0].weight, torch.ones_like(model[0].weight)))
        self.assertTrue(torch.allclose(model[2].weight, torch.ones_like(model[2].weight)))


if __name__ == "__main__":
    unittest.main()