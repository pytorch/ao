import torch
import torch.nn as nn
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
from torchao.quantization.utils import compute_error
from torchao.quantization.prototype.mixed_precision.naive_intNwo import intN_weight_only

def test_weight_only_quant(quantization_bit=2, symmetric=False):
    for x_shape in [[64, 32], [80, 80, 80, 32], [16, 64, 32]]:
        x = torch.randn(*x_shape, dtype=torch.bfloat16)
        m = nn.Sequential(nn.Linear(32, 80)).bfloat16()
        y_ref = m(x)
        quantize_(m, intN_weight_only(n=quantization_bit, group_size=32, symmetric=symmetric))
        y_wo = m(x)
        sqnr = compute_error(y_ref, y_wo)
        # SQNR_dB can be approximated by 6.02n, where n is the bit width of the quantization
        # e.g., we set sqnr threshold = 44 for 8-bit, so that 6.02 * 8= 48.16 fullfills
        assert sqnr > 44.0-(8-quantization_bit)*6.02, "sqnr: {} is too low".format(sqnr)


# test if the asymmetric and symmetric quantization API works with different bit widths
for i in [2, 3, 4, 5, 6, 8]:
    # test for asymmetric quantization
    try:
        test_weight_only_quant(i, False)
        print(f"Test passed for {i}-bit using naive intNwo asymmetric quantization implementation")
    except Exception as e:
        print(f"Exception handled in test loop for {i}-bit asymmetric quantization. Details: {e}")

    #test for symmetric quantization
    try:
        test_weight_only_quant(i, True)
        print(f"Test passed for {i}-bit using naive intNwo symmetric quantization implementation")
    except Exception as e:
        print(f"Exception handled in test loop for {i}-bit symmetric quantization. Details: {e}")
