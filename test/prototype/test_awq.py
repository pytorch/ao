# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import tempfile

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.dtypes import Int4CPULayout
from torchao.prototype.awq import AWQConfig, AWQStep
from torchao.quantization import FbgemmConfig, Int4WeightOnlyConfig, quantize_
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_6,
    _is_fbgemm_genai_gpu_available,
)


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
    and _is_fbgemm_genai_gpu_available()
    and TORCH_VERSION_AT_LEAST_2_6
):
    devices.append("cuda")


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

    @parameterized.expand([(device,) for device in devices])
    def test_awq_functionality(self, device):
        dataset_size = 100
        l1, l2, l3 = 512, 256, 128
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs
        group_size = 128
        n_calibration_examples = 10
        sequence_length = 5

        m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)

        # baseline quantization
        if device == "cuda":
            base_config = FbgemmConfig(
                input_dtype=torch.bfloat16,
                weight_dtype=torch.int4,
                output_dtype=torch.bfloat16,
                block_size=[1, group_size],
                preshuffle=False,
            )
        elif device == "cpu":
            base_config = Int4WeightOnlyConfig(
                group_size=group_size, layout=Int4CPULayout(), set_inductor_config=False
            )
            torch.manual_seed(1234)
        else:
            assert False, "Unsupported device: {}".format(device)
        m_baseline = copy.deepcopy(m)
        quantize_(m_baseline, base_config)

        # awq quantization
        dataset = m.example_inputs(
            dataset_size,
            sequence_length=sequence_length,
            dtype=original_dtype,
            device=device,
        )
        ref_out = torch.cat([m(d.squeeze(0)) for d in dataset])

        calibration_data = dataset[:n_calibration_examples]

        quant_config = AWQConfig(base_config, step=AWQStep.PREPARE)
        quantize_(m, quant_config)

        for example in calibration_data:
            m(example)

        quant_config = AWQConfig(base_config, step=AWQStep.CONVERT)
        quantize_(m, quant_config)

        awq_out = torch.cat([m(d.squeeze(0)) for d in dataset])
        baseline_out = torch.cat([m_baseline(d.squeeze(0)) for d in dataset])

        loss_awq = (ref_out - awq_out).pow(2).mean().item()
        loss_base = (ref_out - baseline_out).pow(2).mean().item()
        assert loss_awq < loss_base

    @parameterized.expand([(device,) for device in devices])
    def test_awq_loading(self, device):
        dataset_size = 100
        l1, l2, l3 = 512, 256, 128
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs
        group_size = 128
        n_calibration_examples = 10
        sequence_length = 5

        m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
        dataset = m.example_inputs(
            dataset_size,
            sequence_length=sequence_length,
            dtype=original_dtype,
            device=device,
        )
        calibration_data = dataset[:n_calibration_examples]

        # calibrate
        if device == "cuda":
            base_config = FbgemmConfig(
                input_dtype=torch.bfloat16,
                weight_dtype=torch.int4,
                output_dtype=torch.bfloat16,
                block_size=[1, group_size],
                preshuffle=False,
            )
        elif device == "cpu":
            base_config = Int4WeightOnlyConfig(
                group_size=group_size, layout=Int4CPULayout(), set_inductor_config=False
            )
        else:
            assert False, "Unsupported device: {}".format(device)
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

        loaded_model = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
        loaded_model.load_state_dict(state_dict, assign=True)

        m = torch.compile(m, fullgraph=True)
        loaded_model = torch.compile(loaded_model, fullgraph=True)

        awq_out = torch.cat([m(d.squeeze(0)) for d in dataset])
        awq_save_load_out = torch.cat([loaded_model(d.squeeze(0)) for d in dataset])

        assert awq_out is not None
        assert awq_save_load_out is not None
        assert torch.allclose(awq_out, awq_save_load_out, atol=1e-2)

    @parameterized.expand([(device,) for device in devices])
    def test_awq_loading_vllm(self, device):
        """Simulate weight loading in vllm:
        * prepare model weight to the same format (awq weight)
        * use weight.copy_(state_dict["weight"]) to copy over the quantized weights from checkpoint

        There is also a slicing op that is ommitted here, overall e2e is tested in tests in vllm repo
        """
        dataset_size = 100
        l1, l2, l3 = 512, 256, 128
        original_dtype = torch.bfloat16  # tinygemm kernel only uses bfloat16 inputs
        group_size = 128
        n_calibration_examples = 10
        sequence_length = 5

        m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
        dataset = m.example_inputs(
            dataset_size,
            sequence_length=sequence_length,
            dtype=original_dtype,
            device=device,
        )
        calibration_data = dataset[:n_calibration_examples]

        # calibrate
        if device == "cuda":
            base_config = FbgemmConfig(
                input_dtype=torch.bfloat16,
                weight_dtype=torch.int4,
                output_dtype=torch.bfloat16,
                block_size=[1, group_size],
                preshuffle=False,
            )
        elif device == "cpu":
            base_config = Int4WeightOnlyConfig(
                group_size=group_size, layout=Int4CPULayout(), set_inductor_config=False
            )
        else:
            assert False, "Unsupported device: {}".format(device)
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

        loaded_model = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
        quant_config = AWQConfig(base_config, step=AWQStep.PREPARE_FOR_LOADING)
        quantize_(loaded_model, quant_config)

        loaded_model.linear1.weight.copy_(state_dict["linear1.weight"])
        loaded_model.linear2.weight.copy_(state_dict["linear2.weight"])
        loaded_model.linear3.weight.copy_(state_dict["linear3.weight"])

        m = torch.compile(m, fullgraph=True)
        loaded_model = torch.compile(loaded_model, fullgraph=True)

        awq_out = torch.cat([m(d.squeeze(0)) for d in dataset])
        awq_save_load_out = torch.cat([loaded_model(d.squeeze(0)) for d in dataset])

        assert awq_out is not None
        assert awq_save_load_out is not None
        assert torch.allclose(awq_out, awq_save_load_out, atol=1e-2)


if __name__ == "__main__":
    run_tests()
