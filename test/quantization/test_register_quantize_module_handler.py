
import unittest
import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import quantize_
from torchao.quantization.transform_module import register_quantize_module_handler


class ModuleSwapConfig(AOBaseConfig):
    """Configuration for testing module swapping functionality"""
    pass


@register_quantize_module_handler(ModuleSwapConfig)
def _module_swap_transform(module, config):
    """Transform function that swaps the module with a new one"""
    # Create a new module with modified weights
    new_module = nn.Linear(module.in_features, module.out_features)
    # Set all weights to ones to make it easy to verify the swap
    new_module.weight = torch.nn.Parameter(torch.ones_like(module.weight))
    if module.bias is not None:
        new_module.bias = torch.nn.Parameter(torch.ones_like(module.bias))
    return new_module


class TestRegisterQuantizeModuleHandler(unittest.TestCase):
    """Test cases for register_quantize_module_handler functionality"""
    
    def test_top_level_module_swap(self):
        """Test that module swapping works for top-level modules"""
        # Create a linear module
        model = nn.Linear(10, 5)
        original_weight = model.weight.clone()
        
        # Apply quantization that swaps the module
        quantize_(model, ModuleSwapConfig())
        
        # Check that the module was actually swapped
        # The weights should now be all ones
        self.assertFalse(torch.equal(model.weight, original_weight))
        self.assertTrue(torch.allclose(model.weight, torch.ones_like(model.weight)))
        
    def test_nested_module_swap(self):
        """Test that module swapping works for nested modules"""
        # Create a sequential model with multiple linear layers
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        original_weight1 = model[0].weight.clone()
        original_weight2 = model[2].weight.clone()
        
        # Apply quantization that swaps the modules
        quantize_(model, ModuleSwapConfig())
        
        # Check that the modules were actually swapped
        self.assertFalse(torch.equal(model[0].weight, original_weight1))
        self.assertFalse(torch.equal(model[2].weight, original_weight2))
        self.assertTrue(torch.allclose(model[0].weight, torch.ones_like(model[0].weight)))
        self.assertTrue(torch.allclose(model[2].weight, torch.ones_like(model[2].weight)))


if __name__ == "__main__":
    unittest.main()