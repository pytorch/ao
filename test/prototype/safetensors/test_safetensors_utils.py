import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao import quantize_
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from torchao.prototype.safetensors.safetensors_utils import is_metadata_torchao
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig
from torchao.utils import (
    is_sm_at_least_89,
)


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
class TestSafeTensorsUtils(TestCase):
    @parametrize(
        "metadata",
        [
            {},  # not metadata
            {"format": "pt"},  # "tensor_names" not in metadata
            {
                "tensor_names": "PerRow()"
            },  # json.loads() fails for metadata["tensor_names"]
            {"tensor_names": []},  # not tensor_names
            {"tensor_names": "0"},  # tensor_names not a list
            {
                "tensor_names": '["0.weight", "0.bias"]',  # tensor_name not in metadata
            },
            {
                "0.weight": {"_type": 0},  # tensor data not str
                "tensor_names": '["0.weight", "0.bias"]',
            },
            {
                "0.weight": "PerRow()",  # json.loads() fails for metadata[tensor_name]
                "tensor_names": '["0.weight", "0.bias"]',
            },
            {
                "0.weight": "{}",  # missing _type key
                "tensor_names": '["0.weight", "0.bias"]',
            },
            {
                "0.weight": '{"_type": "Int4Tensor_NOT_SUPPORTED"}',  # _type not in ALLOWED_TENSORS
                "tensor_names": '["0.weight", "0.bias"]',
            },
        ],
    )
    def test_not_metadata_torchao(self, metadata):
        assert not is_metadata_torchao(metadata)

    def test_metadata_torchao(self):
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 256, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(model, config)
        _, metadata = flatten_tensor_state_dict(model.state_dict())
        assert is_metadata_torchao(metadata)


instantiate_parametrized_tests(TestSafeTensorsUtils)

if __name__ == "__main__":
    run_tests()
