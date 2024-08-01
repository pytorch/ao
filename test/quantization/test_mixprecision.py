import unittest

import torch
import torch.nn as nn
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
from torchao.quantization.utils import compute_error
from torchao.quantization.prototype.mixed_precision.naive_intNwo import intN_weight_only

_CUDA_IS_AVAILABLE = torch.cuda.is_available()

class TestWeightOnlyQuantNaive(unittest.TestCase):

    def test_quantization_2_3_5_6_8_bit(self):
        for quantization_bit in [2, 3, 5, 6, 8]:
            for symmetric in [False, True]:
                with self.subTest(quantization_bit=quantization_bit, symmetric=symmetric):
                    for x_shape in [[64, 32], [80, 80, 80, 32], [16, 64, 32]]:
                        x = torch.randn(*x_shape, dtype=torch.bfloat16)
                        m = nn.Sequential(nn.Linear(32, 80)).bfloat16()
                        y_ref = m(x)
                        quantize_(m, intN_weight_only(n=quantization_bit, group_size=32, symmetric=symmetric))
                        y_wo = m(x)
                        sqnr = compute_error(y_ref, y_wo)
                        expected_sqnr_threshold = 44.0 - (8 - quantization_bit) * 6.02
                        self.assertGreater(sqnr, expected_sqnr_threshold, f"sqnr: {sqnr} is too low")
    '''
    @unittest.skipIf(not _CUDA_IS_AVAILABLE, "skipping int4_wight_only when cuda is not available")
    def test_quantization_4_bit(self):
        for x_shape in [[64, 32], [80, 80, 80, 32], [16, 64, 32]]:
            x = torch.randn(*x_shape, dtype=torch.bfloat16)
            m = nn.Sequential(nn.Linear(32, 80)).bfloat16()
            y_ref = m(x)
            quantize_(m, intN_weight_only(n=4, group_size=32))
            y_wo = m(x)
            sqnr = compute_error(y_ref, y_wo)
            expected_sqnr_threshold = 44.0 - (8 - 4) * 6.02
            self.assertGreater(sqnr, expected_sqnr_threshold, f"sqnr: {sqnr} is too low")
    '''
if __name__ == '__main__':
    unittest.main()
    
'''
def test_weight_only_quant_naive(quantization_bit=2, symmetric=False):
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

for i in [2, 3, 4, 5, 6, 8]:
    # test for asymmetric quantization
    try:
        test_weight_only_quant_naive(i, False)
        print(f"Test passed for {i}-bit using naive intNwo asymmetric quantization implementation")
    except Exception as e:
        print(f"Exception handled in test loop for {i}-bit asymmetric quantization. Details: {e}")

    # test for symmetric quantization
    try:
        test_weight_only_quant_naive(i, True)
        print(f"Test passed for {i}-bit using naive intNwo symmetric quantization implementation")
    except Exception as e:
        print(f"Exception handled in test loop for {i}-bit symmetric quantization. Details: {e}")
'''
