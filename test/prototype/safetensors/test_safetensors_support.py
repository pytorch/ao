import json
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao import quantize_
from torchao.prototype.safetensors.safetensors_support import (
    flatten_tensor_state_dict,
    unflatten_tensor_state_dict,
)
from torchao.quantization.granularity import PerRow
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
)
from torchao.utils import is_sm_at_least_89


def load_data(file_path: str, device: str):
    loaded_tensors = load_file(file_path, device)
    with open(file_path, "rb") as f:
        import struct

        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        metadata = header.get("__metadata__", {})
    return loaded_tensors, metadata


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_89(), "Need sm89+")
class TestSafeTensors(TestCase):
    @parametrize(
        "config, act_pre_scale",
        [
            (Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), False),
            (Int4WeightOnlyConfig(), False),
            (Int4WeightOnlyConfig(), True),
            (Int4WeightOnlyConfig(int4_packing_format="tile_packed_to_4d"), True),
            (Float8DynamicActivationInt4WeightConfig(), False),
        ],
    )
    def test_safetensors(self, config, act_pre_scale=False):
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(model, config)
        if act_pre_scale:
            model[0].weight.act_pre_scale = torch.ones(
                (1), dtype=torch.bfloat16, device="cuda"
            )
        example_inputs = (torch.randn(2, 128, dtype=torch.bfloat16, device="cuda"),)
        ref_output = model(*example_inputs)

        with tempfile.NamedTemporaryFile() as f:
            tensors_data_dict, metadata = flatten_tensor_state_dict(model.state_dict())
            save_file(tensors_data_dict, f.name, metadata=metadata)
            tensors_data_dict, metadata = load_data(file_path=f.name, device="cuda")
            reconstructed_dict = unflatten_tensor_state_dict(
                tensors_data_dict, metadata
            )

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        )
        model.load_state_dict(reconstructed_dict, assign=True)
        output = model(*example_inputs)
        assert torch.equal(output, ref_output)


instantiate_parametrized_tests(TestSafeTensors)

if __name__ == "__main__":
    run_tests()
