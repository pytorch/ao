import torch
import torch.nn as nn

from naive_intNwo import intN_weight_only_asym, intN_weight_only_sym

from torchao.quantization import quantize_

from torchao.quantization.utils import (
    _apply_logging_hook,
    compute_error,
    compute_error as SQNR,
    _fqn_to_op_to_shape_to_count,
    LoggingTensorMode,
)

def test_weight_only_quant(quantization_bit=2):
    for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
        x = torch.randn(*x_shape)
        m = nn.Sequential(nn.Linear(4, 5))
        y_ref = m(x)
        quantize_(m, intN_weight_only_asym(n=int(quantization_bit),group_size=2))
        y_wo = m(x)
        sqnr = compute_error(y_ref, y_wo)
        assert(sqnr > 44.0),"sqnr: {} is too low".format(sqnr)

for i in [2,3,5,6]:
    test_weight_only_quant(i)
