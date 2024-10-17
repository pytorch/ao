from copy import deepcopy
import os
import pytest
import torch
import tempfile
from torch.ao.quantization import HistogramObserver
from torchao.quantization import quantize_
from torchao.utils import (
  TORCH_VERSION_AT_LEAST_2_2,
  TORCH_VERSION_AT_LEAST_2_4,
)
from torchao.quantization.utils import (
  dynamically_quantize_per_channel,
  dequantize_per_channel,
)
from torchao.prototype.smoothquant import (
    insert_smooth_quant_observer,
    smooth_quant,
    SmoothQuantObservedLinear,
    save_smooth_quant_recipe,
    load_smooth_quant_recipe
)

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 1, bias=False)

    def example_inputs(self, batch_size, sequence_length=10, dtype=torch.bfloat16, device="cuda"):
        return [torch.randn(1, sequence_length, self.linear1.in_features, dtype=dtype, device=device) for j in range(batch_size)]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


bias_list = [True, False]
alpha_list = [0.5, 0.75]
quant_mode_list = ["static", "dynamic"]
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
idtypes = (torch.float, torch.bfloat16, torch.half)

@pytest.mark.parametrize("bias", bias_list)
@pytest.mark.parametrize("alpha", alpha_list)
@pytest.mark.parametrize("quant_mode", quant_mode_list)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("idtype", idtypes)
def test_compute(bias, alpha, quant_mode, device, idtype):
    class Linear(torch.nn.Module):
        def __init__(self, bias: bool):
            super().__init__()
            self.fc = torch.nn.Linear(16, 16, bias)
            self.fc.weight.data = torch.randn_like(self.fc.weight.data)

        def forward(self, x):
            return self.fc(x)

    m = Linear(bias).eval().to(idtype).to(device)
    m_ref = deepcopy(m)
    data = torch.randn(2, 16, dtype=idtype, device=device)

    # calibrate
    insert_smooth_quant_observer(m, alpha, quant_mode, n_calib_examples=1)
    m(data)
    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
    quantize_(m, smooth_quant(), is_observed_linear)
    with torch.inference_mode():
        # m = torch.compile(m, fullgraph=True)
        out = m(data)

        # reference
        weight = m_ref.fc.weight.data.float()
        b = m_ref.fc.bias if bias else None
        x_abs_max_per_ic = torch.abs(data).max(dim=0).values
        w_abs_max_per_ic = torch.abs(weight).max(dim=0).values
        smoothing_factor = (
            torch.pow(x_abs_max_per_ic, alpha) / torch.pow(
            w_abs_max_per_ic, 1 - alpha)
        )
        act = data / smoothing_factor
        wei = weight * smoothing_factor
        qw, w_scales, w_zps = dynamically_quantize_per_channel(
            wei, -127, 127, torch.int8
        )
        fq_wei = dequantize_per_channel(qw, w_scales, w_zps, idtype)
        if (device == "cpu" and not TORCH_VERSION_AT_LEAST_2_4) or \
            not TORCH_VERSION_AT_LEAST_2_2:
            # _int_mm is not supported in these cases
            out_ref = torch.nn.functional.linear(act.to(idtype), fq_wei, b)
        elif quant_mode == "static":
            # activation is quantized per-tensor
            obs = HistogramObserver(
                dtype=torch.int8,
                qscheme=torch.per_tensor_symmetric,
                quant_min=-127,
                quant_max=127,
            )
            obs(act.float().to("cpu"))
            act_scale, _ = obs.calculate_qparams()
            fq_act = torch.quantize_per_tensor(
                act.float(), scale=act_scale.item(), zero_point=0, dtype=torch.qint8
            ).dequantize().to(idtype)
            out_ref = torch.nn.functional.linear(fq_act, fq_wei, b)
        else:
            # activation is quantized per-row (batch * sequence_length)
            qx, x_scales, x_zps = dynamically_quantize_per_channel(
                act.float(), -127, 127, torch.int8
            )
            fq_act = dequantize_per_channel(qx, x_scales, x_zps, idtype)
            out_ref = torch.nn.functional.linear(fq_act, fq_wei, b)

        # BFloat16 and Float16 have larger errors
        assert torch.allclose(out, out_ref.to(idtype), atol = 0.2)


@pytest.mark.parametrize("alpha", alpha_list)
@pytest.mark.parametrize("quant_mode", quant_mode_list)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("idtype", idtypes)
def test_save_load_recipe(alpha, quant_mode, device, idtype):
    dataset_size = 20
    l1, l2, l3 = 512, 256, 128
    original_dtype = idtype
    n_calib_examples = 10
    sequence_length = 5

    m = ToyLinearModel(l1, l2, l3).eval().to(original_dtype).to(device)
    m_save_load = deepcopy(m)

    dataset = m.example_inputs(dataset_size, sequence_length=sequence_length, dtype=original_dtype, device=device)
    calibration_data = dataset[:n_calib_examples]

    # calibrate
    insert_smooth_quant_observer(m, alpha, quant_mode, n_calib_examples=n_calib_examples)
    insert_smooth_quant_observer(m_save_load, alpha, quant_mode, n_calib_examples=n_calib_examples)

    for example in calibration_data:
        m(example.to(device))
        m_save_load(example.to(device))

    with tempfile.NamedTemporaryFile() as fp:
        save_path = fp.name
        save_smooth_quant_recipe(m_save_load, save_path)
        load_smooth_quant_recipe(m_save_load, save_path)

    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
    quantize_(m, smooth_quant(), is_observed_linear)
    # m = torch.compile(m, fullgraph=True)
    # m_save_load = torch.compile(m_save_load, fullgraph=True)
    out_list = [m(data.squeeze(0)) for data in dataset]
    out = torch.cat(out_list)
    save_load_out_list = [m_save_load(data.squeeze(0)) for data in dataset]
    save_load_out = torch.cat(save_load_out_list)
    
    assert out is not None
    assert save_load_out is not None
    assert torch.allclose(out, save_load_out)