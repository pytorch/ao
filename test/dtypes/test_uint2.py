from unittest import main

import torch
import torch.nn as nn

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
)

from torchao.dtypes.uint2 import (
    BitnetTensor
)
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
)

def _apply_weight_only_uint2_quant(model):
    def fn(mod):
        mod.weight = torch.nn.Parameter(BitnetTensor.from_float(mod.weight), requires_grad=False)
        return mod

    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: fn(mod),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )


class TestUInt2(QuantizationTestCase):
    def test_gpu_quant(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
            x = torch.randn(*x_shape).to(device)
            m = nn.Sequential(nn.Linear(4, 16)).to(device)
            y_ref = m(x)
            _apply_weight_only_uint2_quant(m)
            y_wo = m(x)
            # sqnr = compute_error(y_ref, y_wo)
            # opt = torch.compile(m, fullgraph=True, mode="max-autotune")
            # make sure it runs
            # opt(x)


if __name__ == "__main__":
    main()
