"""
Quick test to verify SINQ_SCALE_ONLY is properly linked to the API
"""
import torch
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.utils import compute_error


def test_sinq_intx_weight_only_config():
    print("Testing SINQ with IntxWeightOnlyConfig...")
    dtype = torch.bfloat16
    device = "cpu"
    config = IntxWeightOnlyConfig(
        weight_dtype=torch.int4,
        granularity=PerGroup(32),
        intx_choose_qparams_algorithm="sinq_scale_only",
    )
    input_tensor = torch.randn(1, 128, dtype=dtype, device=device)
    linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
    original = linear(input_tensor)
    quantize_(linear, config)
    quantized = linear(input_tensor)
    error = compute_error(original, quantized)
    print(f"  Error: {error:.2f} dB")
    assert error > 20, f"Got error {error}, expected > 20 dB"
    print("  PASSED!")


def test_sinq_int8_dyn_act_intx_weight_config():
    print("Testing SINQ with Int8DynamicActivationIntxWeightConfig...")
    dtype = torch.bfloat16
    device = "cpu"
    config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(32),
        intx_choose_qparams_algorithm="sinq_scale_only",
    )
    input_tensor = torch.randn(1, 128, dtype=dtype, device=device)
    linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
    original = linear(input_tensor)
    quantize_(linear, config)
    quantized = linear(input_tensor)
    error = compute_error(original, quantized)
    print(f"  Error: {error:.2f} dB")
    assert error > 20, f"Got error {error}, expected > 20 dB"
    print("  PASSED!")


def test_enum_access():
    print("Testing enum access...")
    from torchao.quantization.quantize_.workflows import IntxChooseQParamsAlgorithm

    print(f"  SINQ_SCALE_ONLY: {IntxChooseQParamsAlgorithm.SINQ_SCALE_ONLY}")
    print(f"  HQQ_SCALE_ONLY: {IntxChooseQParamsAlgorithm.HQQ_SCALE_ONLY}")
    print(f"  AFFINE: {IntxChooseQParamsAlgorithm.AFFINE}")
    assert IntxChooseQParamsAlgorithm.SINQ_SCALE_ONLY.value == "sinq_scale_only"
    print("  PASSED!")


if __name__ == "__main__":
    test_enum_access()
    test_sinq_intx_weight_only_config()
    test_sinq_int8_dyn_act_intx_weight_config()
    print("\nAll tests passed!")
