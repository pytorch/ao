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
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
)
from torchao.utils import get_available_devices, is_sm_at_least_89

_DEVICES = get_available_devices()
_DEVICE = _DEVICES[-1]
assert _DEVICE in ["cuda", "xpu"], "Test currently only supports CUDA & XPU"


_TEST_CONFIGS = [
    (Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), False),
    (Int4WeightOnlyConfig(int4_packing_format="plain_int32"), False),
    (IntxWeightOnlyConfig(), False),
    (Int8DynamicActivationIntxWeightConfig(), False),
    (Int8WeightOnlyConfig(version=2), False),
    (Int8DynamicActivationInt8WeightConfig(version=2), False),
]

# Build test configs - CUDA supports all Int4 packing formats, XPU only supports plain_int32.
if _DEVICE == "cuda":
    _TEST_CONFIGS.extend(
        [
            (Int4WeightOnlyConfig(), False),
            (Int4WeightOnlyConfig(), True),
        ]
    )


def load_data(file_path: str, device: str):
    loaded_tensors = load_file(file_path, device)
    with open(file_path, "rb") as f:
        import struct

        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        metadata = header.get("__metadata__", {})
    return loaded_tensors, metadata


@unittest.skipIf(
    not torch.cuda.is_available() and not torch.xpu.is_available(),
    "Need CUDA or XPU available",
)
@unittest.skipIf(
    torch.cuda.is_available() and not is_sm_at_least_89(), "Need sm89+ for CUDA"
)
class TestSafeTensors(TestCase):
    @parametrize("config, act_pre_scale", _TEST_CONFIGS)
    def test_safetensors(self, config, act_pre_scale=False):
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=_DEVICE)
        )
        quantize_(model, config)
        if act_pre_scale:
            model[0].weight.act_pre_scale = torch.ones(
                (1), dtype=torch.bfloat16, device=_DEVICE
            )
        example_inputs = (torch.randn(2, 128, dtype=torch.bfloat16, device=_DEVICE),)
        ref_output = model(*example_inputs)

        with tempfile.NamedTemporaryFile() as f:
            tensors_data_dict, metadata = flatten_tensor_state_dict(model.state_dict())

            for key in tensors_data_dict.keys():
                assert key.startswith("0._weight_") or key.startswith("0.bias"), (
                    f"Unexpected key format: {key}"
                )

            save_file(tensors_data_dict, f.name, metadata=metadata)
            tensors_data_dict, metadata = load_data(file_path=f.name, device=_DEVICE)
            reconstructed_dict, leftover_tensor_data_dict = unflatten_tensor_state_dict(
                tensors_data_dict, metadata
            )
            assert not leftover_tensor_data_dict

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=_DEVICE)
        )
        model.load_state_dict(reconstructed_dict, assign=True)
        output = model(*example_inputs)
        assert torch.equal(output, ref_output)

    @parametrize("config, act_pre_scale", _TEST_CONFIGS)
    def test_safetensors_sharded(self, config, act_pre_scale=False):
        print("config is ", config)
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=_DEVICE)
        )
        quantize_(model, config)
        if act_pre_scale:
            model[0].weight.act_pre_scale = torch.ones(
                (1), dtype=torch.bfloat16, device=_DEVICE
            )

        with tempfile.NamedTemporaryFile() as f:
            tensors_data_dict, metadata = flatten_tensor_state_dict(model.state_dict())
            save_file(tensors_data_dict, f.name, metadata=metadata)
            tensors_data_dict, metadata = load_data(file_path=f.name, device=_DEVICE)

            # simulate missing info on future file
            if act_pre_scale:
                del tensors_data_dict["0._weight_act_pre_scale"]  # optional tensor data
            else:
                del tensors_data_dict["0._weight_qdata"]

            reconstructed_dict, leftover_tensor_data_dict = unflatten_tensor_state_dict(
                tensors_data_dict, metadata
            )

            # since qdata is missing, layer 0 should not have been processed
            for key in tensors_data_dict.keys():
                if key.startswith("0._weight_"):
                    assert key in leftover_tensor_data_dict


instantiate_parametrized_tests(TestSafeTensors)

if __name__ == "__main__":
    run_tests()
