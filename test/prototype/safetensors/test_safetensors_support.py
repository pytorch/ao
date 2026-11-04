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
from torchao.prototype.mx_formats.inference_workflow import (
    MXDynamicActivationMXWeightConfig,
    NVFP4DynamicActivationNVFP4WeightConfig,
)
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
from torchao.utils import (
    get_available_devices,
    is_sm_at_least_89,
    is_sm_at_least_100,
    torch_version_at_least,
)

_ALL_TEST_CONFIGS = [
    (Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), False),
    (IntxWeightOnlyConfig(), False),
    (Int8DynamicActivationIntxWeightConfig(), False),
    (Int8WeightOnlyConfig(version=2), False),
    (Int8DynamicActivationInt8WeightConfig(version=2), False),
]

# plain_int32 only supports XPU
if torch.xpu.is_available():
    _ALL_TEST_CONFIGS += [
        (Int4WeightOnlyConfig(int4_packing_format="plain_int32"), False),
    ]
else:
    # Int4WeightOnlyConfig with tinygemm uses sm90a CUTLASS kernels which are not
    # forward-compatible with sm100+
    if not is_sm_at_least_100():
        _ALL_TEST_CONFIGS += [
            (Int4WeightOnlyConfig(), False),
            (Int4WeightOnlyConfig(), True),
            (Int4WeightOnlyConfig(int4_packing_format="tile_packed_to_4d"), False),
        ]

    # MX and NVFP4 configs require torch >= 2.11
    if torch_version_at_least("2.11.0.dev"):
        _ALL_TEST_CONFIGS += [
            (MXDynamicActivationMXWeightConfig(), False),
            (NVFP4DynamicActivationNVFP4WeightConfig(), False),
        ]

_TEST_CONFIGS = _ALL_TEST_CONFIGS


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
    @parametrize("device", get_available_devices())
    def test_safetensors(self, config, device, act_pre_scale=False):
        if device == "cpu":
            self.skipTest("Need GPU available")
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
        )
        quantize_(model, config)
        if act_pre_scale:
            model[0].weight.act_pre_scale = torch.ones(
                (1), dtype=torch.bfloat16, device=device
            )
        example_inputs = (torch.randn(128, 128, dtype=torch.bfloat16, device=device),)
        ref_output = model(*example_inputs)

        with tempfile.NamedTemporaryFile() as f:
            tensors_data_dict, metadata = flatten_tensor_state_dict(model.state_dict())

            for key in tensors_data_dict.keys():
                assert key.startswith("0._weight_") or key.startswith("0.bias"), (
                    f"Unexpected key format: {key}"
                )

            save_file(tensors_data_dict, f.name, metadata=metadata)
            tensors_data_dict, metadata = load_data(file_path=f.name, device=device)
            reconstructed_dict, leftover_tensor_data_dict = unflatten_tensor_state_dict(
                tensors_data_dict, metadata
            )
            assert not leftover_tensor_data_dict

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
        )
        model.load_state_dict(reconstructed_dict, assign=True)
        output = model(*example_inputs)
        assert torch.equal(output, ref_output)

    @parametrize("config, act_pre_scale", _TEST_CONFIGS)
    @parametrize("device", get_available_devices())
    def test_safetensors_sharded(self, config, device, act_pre_scale=False):
        if device == "cpu":
            self.skipTest("Need GPU available")
        print("config is ", config)
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256, dtype=torch.bfloat16, device=device)
        )
        quantize_(model, config)
        if act_pre_scale:
            model[0].weight.act_pre_scale = torch.ones(
                (1), dtype=torch.bfloat16, device=device
            )

        with tempfile.NamedTemporaryFile() as f:
            tensors_data_dict, metadata = flatten_tensor_state_dict(model.state_dict())
            save_file(tensors_data_dict, f.name, metadata=metadata)
            tensors_data_dict, metadata = load_data(file_path=f.name, device=device)

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


@unittest.skipIf(
    not torch.backends.mps.is_available(),
    "Need MPS available",
)
class TestSafeTensorsMPS(TestCase):
    """Test safetensors serialization for IntxMPSExperimentalTensor on MPS."""

    def test_intx_mps_experimental_tensor(self):
        import json
        import struct

        from torchao.experimental.ops.mps.utils import _load_torchao_mps_lib
        from torchao.prototype.quantization.intx_mps.intx_mps_experimental_tensor import (
            IntxMPSExperimentalTensor,
        )

        _load_torchao_mps_lib()

        device = "mps"
        N, K, group_size, nbit = 128, 256, 64, 4
        packed = torch.randint(
            0, 255, (N, nbit * K // 8), dtype=torch.uint8, device=device
        )
        scales = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
        zeros = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
        tensor = IntxMPSExperimentalTensor(
            packed, scales, zeros, nbit, [1, group_size], torch.Size([N, K])
        )
        state_dict = {"0.weight": tensor}

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            tensors_data_dict, metadata = flatten_tensor_state_dict(state_dict)
            save_file(tensors_data_dict, f.name, metadata=metadata)
            loaded_tensors = load_file(f.name, device=device)
            with open(f.name, "rb") as fh:
                header_size = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(header_size))
            loaded_metadata = header.get("__metadata__", {})
            reconstructed, leftover = unflatten_tensor_state_dict(
                loaded_tensors, loaded_metadata
            )
            assert not leftover, f"Leftover tensors: {leftover}"

            rt = reconstructed["0.weight"]
            assert isinstance(rt, IntxMPSExperimentalTensor)
            assert rt.nbit == nbit
            assert rt.block_size == [1, group_size]
            assert list(rt.shape) == [N, K]
            assert torch.equal(rt.packed_weight, packed)
            assert torch.equal(rt.scales, scales)
            assert torch.equal(rt.zeros, zeros)


instantiate_parametrized_tests(TestSafeTensors)
instantiate_parametrized_tests(TestSafeTensorsMPS)

if __name__ == "__main__":
    run_tests()
