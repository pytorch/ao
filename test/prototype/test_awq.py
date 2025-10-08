# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import tempfile

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.prototype.awq import AWQConfig, AWQStep
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.utils import _is_fbgemm_gpu_genai_available, torch_version_at_least


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 64, bias=False)

    def example_inputs(
        self, batch_size, sequence_length=10, dtype=torch.bfloat16, device="cuda"
    ):
        return [
            torch.randn(
                1, sequence_length, self.linear1.in_features, dtype=dtype, device=device
            )
            for j in range(batch_size)
        ]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


devices = ["cpu"]
if (
    torch.cuda.is_available()
    and _is_fbgemm_gpu_genai_available()
    and torch_version_at_least("2.6.0")
):
    devices.append("cuda")


if torch.xpu.is_available():
    devices.append("xpu")


device_to_base_configs = {
    "cuda": [
        Int4WeightOnlyConfig(group_size=128),
        # Note: the functionality unit test doesn't work for hqq
        Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d"),
    ],
    "cpu": [Int4WeightOnlyConfig(group_size=128, int4_packing_format="opaque")],
    "xpu": [Int4WeightOnlyConfig(group_size=128, int4_packing_format="plain_int32")],
}


class TestAWQ(TestCase):
    def test_awq_config(self):
        base_config = Int4WeightOnlyConfig()
        AWQConfig(base_config, step=AWQStep.PREPARE)
        AWQConfig(base_config, step=AWQStep.PREPARE_FOR_LOADING)
        AWQConfig(base_config, step=AWQStep.CONVERT)

        AWQConfig(base_config, step="prepare")
        AWQConfig(base_config, step="prepare_for_loading")
        AWQConfig(base_config, step="convert")

        with self.assertRaisesRegex(ValueError, "is not one of"):
            AWQConfig(base_config, step="not_supported")

    @parametrize("device", devices)
    def test_awq_functionality(self, device):
        dataset_size = 10
        l1, l2, l3 = 512, 256, 128
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs
        sequence_length = 5

        assert device in device_to_base_configs, "Unsupported device: {}".format(device)
        base_configs = device_to_base_configs[device]

        for base_config in base_configs:
            m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
            m_baseline = copy.deepcopy(m)

            dataset = m.example_inputs(
                dataset_size,
                sequence_length=sequence_length,
                dtype=original_dtype,
                device=device,
            )
            # for test, we use calibration_data = dataset so that awq is
            # guranteed to be better than baseline
            # in reality, calibration_data will be a small subset or a different
            # dataset
            calibration_data = dataset
            # concatenatd inputs
            input_cat = torch.cat(calibration_data, dim=-2)
            ref_out = m(input_cat)

            # baseline quantization
            quantize_(m_baseline, base_config)

            # awq quantization
            quant_config = AWQConfig(base_config, step=AWQStep.PREPARE)
            quantize_(m, quant_config)

            for example in calibration_data:
                m(example)

            quant_config = AWQConfig(base_config, step=AWQStep.CONVERT)
            quantize_(m, quant_config)

            # evaluating on calibration data set to remove any uncertainty
            awq_out = m(input_cat)
            baseline_out = m_baseline(input_cat)

            loss_awq = (ref_out - awq_out).pow(2).mean().item()
            loss_base = (ref_out - baseline_out).pow(2).mean().item()
            assert loss_awq <= loss_base

    @parametrize("device", devices)
    def test_awq_loading(self, device):
        dataset_size = 10
        l1, l2, l3 = 512, 256, 128
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs
        sequence_length = 5

        assert device in device_to_base_configs, "Unsupported device: {}".format(device)
        base_configs = device_to_base_configs[device]

        for base_config in base_configs:
            m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
            dataset = m.example_inputs(
                dataset_size,
                sequence_length=sequence_length,
                dtype=original_dtype,
                device=device,
            )
            # for test purpose, we don't need to get a subset
            calibration_data = dataset
            # concatenatd inputs
            input_cat = torch.cat(calibration_data, dim=-2)

            # calibrate

            quant_config = AWQConfig(base_config, step=AWQStep.PREPARE)
            quantize_(m, quant_config)

            for example in calibration_data:
                m(example)

            # quantize
            quant_config = AWQConfig(base_config, step=AWQStep.CONVERT)
            quantize_(m, quant_config)

            with tempfile.NamedTemporaryFile() as f:
                torch.save(m.state_dict(), f)
                f.seek(0)
                state_dict = torch.load(f)

            loaded_model = (
                ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
            )
            loaded_model.load_state_dict(state_dict, assign=True)

            m = torch.compile(m, fullgraph=True)
            loaded_model = torch.compile(loaded_model, fullgraph=True)

            awq_out = m(input_cat)
            awq_save_load_out = loaded_model(input_cat)

            assert awq_out is not None
            assert awq_save_load_out is not None
            assert torch.allclose(awq_out, awq_save_load_out, atol=1e-2)

    @parametrize("device", devices)
    def test_awq_loading_vllm(self, device):
        """Simulate weight loading in vllm:
        * prepare model weight to the same format (awq weight)
        * use weight.copy_(state_dict["weight"]) to copy over the quantized weights from checkpoint

        There is also a slicing op that is ommitted here, overall e2e is tested in tests in vllm repo
        """
        dataset_size = 10
        l1, l2, l3 = 512, 256, 128
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs
        sequence_length = 5

        assert device in device_to_base_configs, "Unsupported device: {}".format(device)
        base_configs = device_to_base_configs[device]

        for base_config in base_configs:
            m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
            dataset = m.example_inputs(
                dataset_size,
                sequence_length=sequence_length,
                dtype=original_dtype,
                device=device,
            )
            # for test purpose, we don't need to get a subset
            calibration_data = dataset
            # concatenatd inputs
            input_cat = torch.cat(calibration_data, dim=-2)

            # calibrate
            quant_config = AWQConfig(base_config, step=AWQStep.PREPARE)
            quantize_(m, quant_config)

            for example in calibration_data:
                m(example)

            # quantize
            quant_config = AWQConfig(base_config, step=AWQStep.CONVERT)
            quantize_(m, quant_config)

            with tempfile.NamedTemporaryFile() as f:
                torch.save(m.state_dict(), f)
                f.seek(0)
                state_dict = torch.load(f)

            loaded_model = (
                ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
            )
            quant_config = AWQConfig(base_config, step=AWQStep.PREPARE_FOR_LOADING)
            quantize_(loaded_model, quant_config)

            loaded_model.linear1.weight.copy_(state_dict["linear1.weight"])
            loaded_model.linear2.weight.copy_(state_dict["linear2.weight"])
            loaded_model.linear3.weight.copy_(state_dict["linear3.weight"])

            m = torch.compile(m, fullgraph=True)
            loaded_model = torch.compile(loaded_model, fullgraph=True)

            awq_out = m(input_cat)
            awq_save_load_out = loaded_model(input_cat)

            assert awq_out is not None
            assert awq_save_load_out is not None
            assert torch.allclose(awq_out, awq_save_load_out, atol=1e-2)


instantiate_parametrized_tests(TestAWQ)

if __name__ == "__main__":
    run_tests()
