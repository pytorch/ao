import unittest

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
import torch
from torchao import quantize_

from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerRow

from torchao.prototype.quantization.safetensors_support import (
    save_tensor_subclass_dict,
    load_tensor_subclass_dict,
)

@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
class TestSafeTensors(TestCase):
    def test_safetensors(self):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(model, config)
        example_inputs = (torch.randn(2, 32, dtype=torch.bfloat16, device="cuda"),)
        ref_output = model(*example_inputs)

        save_tensor_subclass_dict(model.state_dict(), "fp8_weights.safetensors")
        reconstructed_dict = load_tensor_subclass_dict("fp8_weights.safetensors")

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
        )
        model.load_state_dict(reconstructed_dict, assign=True)
        output = model(*example_inputs)
        assert torch.equal(output, ref_output)


if __name__ == "__main__":
    run_tests()
