# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
from copy import deepcopy

import pytest
import torch

from torchao.prototype.smoothquant import (
    SmoothQuantConfig,
    SmoothQuantObservedLinear,
    insert_smooth_quant_observer_,
    load_smooth_quant_recipe,
    save_smooth_quant_recipe,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import (
    dequantize_per_channel,
    dynamically_quantize_per_channel,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    is_sm_at_least_90,
)

if torch.version.hip is not None:
    pytest.skip("Skipping the test in ROCm", allow_module_level=True)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 1, bias=False)

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


bias_list = [True, False]
alpha_list = [None, 0.5, 0.75]
quant_mode_list = ["static", "dynamic"]
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
idtypes = (torch.float, torch.bfloat16, torch.half)

if TORCH_VERSION_AT_LEAST_2_5:
    # This test case will trigger recompilation many times, so set a large cache_size_limit here
    torch._dynamo.config.cache_size_limit = 128


@pytest.mark.skipif(
    is_sm_at_least_90(), reason="Test failing on H100"
)  # TODO: Fix this test on H100
@pytest.mark.parametrize("bias", bias_list)
@pytest.mark.parametrize("alpha", alpha_list)
@pytest.mark.parametrize("quant_mode", quant_mode_list)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("idtype", idtypes)
@pytest.mark.skip("this test is broken on recent PyTorch, TODO(#1639): fix it")
def test_compute(bias, alpha, quant_mode, device, idtype):
    class Linear(torch.nn.Module):
        def __init__(self, bias: bool):
            super().__init__()
            self.fc = torch.nn.Linear(32, 32, bias)
            self.fc.weight.data = torch.randn_like(self.fc.weight.data)

        def forward(self, x):
            return self.fc(x)

    m = Linear(bias).eval().to(idtype).to(device)
    m_ref = deepcopy(m)
    data = torch.randn(2, 32, dtype=idtype, device=device)

    # calibrate
    insert_smooth_quant_observer_(m, alpha, quant_mode)
    m(data)
    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
    quantize_(m, SmoothQuantConfig(), is_observed_linear)
    with torch.inference_mode():
        if TORCH_VERSION_AT_LEAST_2_5:
            m = torch.compile(m, fullgraph=True)
        out = m(data)

        # reference
        weight = m_ref.fc.weight.data.float()
        b = m_ref.fc.bias if bias else None
        x_abs_max_per_ic = torch.abs(data).max(dim=0).values
        w_abs_max_per_ic = torch.abs(weight).max(dim=0).values
        smoothing_factor = (
            1
            if alpha is None
            else (
                torch.pow(x_abs_max_per_ic, alpha)
                / torch.pow(w_abs_max_per_ic, 1 - alpha)
            )
        )
        act = data / smoothing_factor
        wei = weight * smoothing_factor
        qw, w_scales, w_zps = dynamically_quantize_per_channel(
            wei, -127, 127, torch.int8
        )
        fq_wei = dequantize_per_channel(qw, w_scales, w_zps, idtype)
        if quant_mode == "static":
            # activation is quantized per-tensor
            act_min, act_max = torch.aminmax(act.float())
            max_val_pos = torch.max(-act_min, act_max)
            act_scale = max_val_pos / 127.0
            fq_act = (
                torch.quantize_per_tensor(
                    act.float(), scale=act_scale.item(), zero_point=0, dtype=torch.qint8
                )
                .dequantize()
                .to(idtype)
            )
            out_ref = torch.nn.functional.linear(fq_act, fq_wei, b)
        else:
            # activation is quantized per-row (batch * sequence_length)
            qx, x_scales, x_zps = dynamically_quantize_per_channel(
                act.float(), -127, 127, torch.int8
            )
            fq_act = dequantize_per_channel(qx, x_scales, x_zps, idtype)
            out_ref = torch.nn.functional.linear(fq_act, fq_wei, b)

        # BFloat16 and Float16 have larger errors
        atol = 0.1 if idtype == torch.float else (0.2 if idtype == torch.half else 0.3)
        assert torch.allclose(out, out_ref.to(idtype), atol=atol)


@pytest.mark.skipif(
    is_sm_at_least_90(), reason="Test failing on H100"
)  # TODO: fix this test on H100
@pytest.mark.parametrize("alpha", alpha_list)
@pytest.mark.parametrize("quant_mode", quant_mode_list)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("idtype", idtypes)
@pytest.mark.skip("this test is broken on recent PyTorch, TODO(#1639): fix it")
def test_save_load_recipe(alpha, quant_mode, device, idtype):
    dataset_size = 20
    l1, l2, l3 = 512, 256, 128
    original_dtype = idtype
    n_calib_examples = 10
    sequence_length = 5

    m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
    m_save_load = deepcopy(m)

    dataset = m.example_inputs(
        dataset_size,
        sequence_length=sequence_length,
        dtype=original_dtype,
        device=device,
    )
    calibration_data = dataset[:n_calib_examples]

    # calibrate
    insert_smooth_quant_observer_(m, alpha, quant_mode)
    insert_smooth_quant_observer_(m_save_load, alpha, quant_mode)

    for example in calibration_data:
        m(example.to(device))
        m_save_load(example.to(device))

    with tempfile.NamedTemporaryFile() as fp:
        save_path = fp.name
        save_smooth_quant_recipe(m_save_load, save_path)
        load_smooth_quant_recipe(m_save_load, save_path)

    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
    quantize_(m, SmoothQuantConfig(), is_observed_linear)
    if TORCH_VERSION_AT_LEAST_2_5:
        # earlier versions are not compatible
        m = torch.compile(m, fullgraph=True)
        m_save_load = torch.compile(m_save_load, fullgraph=True)
    out_list = [m(data.squeeze(0)) for data in dataset]
    out = torch.cat(out_list)
    save_load_out_list = [m_save_load(data.squeeze(0)) for data in dataset]
    save_load_out = torch.cat(save_load_out_list)

    assert out is not None
    assert save_load_out is not None
    assert torch.allclose(out, save_load_out)
