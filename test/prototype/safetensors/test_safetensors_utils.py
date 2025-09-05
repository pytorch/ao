import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao import quantize_
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from torchao.prototype.safetensors.safetensors_utils import is_metadata_dict_torchao
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig


class TestSafeTensorsUtils(TestCase):
    @parametrize(
        "metadata_dict",
        [
            {"format": "pt"},
            {
                "0.weight": '{"_type": "Int4Tensor"}',
                "tensor_names": '["0.weight", "0.bias"]',
            },
        ],
    )
    def test_not_metadata_dict_torchao(self, metadata_dict):
        assert not is_metadata_dict_torchao(metadata_dict)

    def test_metadata_dict_torchao(self):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(model, config)
        _, metadata_dict = flatten_tensor_state_dict(model.state_dict())
        assert is_metadata_dict_torchao(metadata_dict)


instantiate_parametrized_tests(TestSafeTensorsUtils)

if __name__ == "__main__":
    run_tests()
