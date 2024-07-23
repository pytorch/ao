import torch
import torch.nn as nn

import os
import sys
# append the path to the naive_intNwo.py file
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "torchao/quantization/prototype/mixed_precision/scripts"))
from naive_intNwo import intN_weight_only

from torchao.quantization import quantize_, int8_weight_only, int4_weight_only

from torchao.quantization.utils import (
    _apply_logging_hook,
    compute_error,
    compute_error as SQNR,
    _fqn_to_op_to_shape_to_count,
    LoggingTensorMode,
)

def test_weight_only_quant(quantization_bit=2, symmetric=False):
    for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
        x = torch.randn(*x_shape)
        m = nn.Sequential(nn.Linear(4, 5))
        y_ref = m(x)
        quantize_(m, intN_weight_only(n=quantization_bit, group_size=2, symmetric=symmetric))
        y_wo = m(x)
        sqnr = compute_error(y_ref, y_wo)
        print(sqnr)
        assert sqnr > 44.0, "sqnr: {} is too low".format(sqnr)


# test if the asymmetric and symmetric quantization API works with different bit widths
for i in range(2, 9):
    #test for asymmetric quantization
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
