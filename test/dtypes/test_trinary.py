import torch
from torchao.dtypes.trinary import (
    TrinaryTensor,
    quantize_per_tensor_trinary,
)
import unittest
from unittest import TestCase, main
from torch._export import capture_pre_autograd_graph
from torch._export import dynamic_dim
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
)
from torchao.quantization.utils import (
    compute_error,
)
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
)
from torch.ao.quantization.observer import ObserverBase
from torch import nn
from torch.fx import (
    Node,
    GraphModule,
)
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
)
import copy

def _quantize_linear_weights_only(model):
    def fn(mod):
        mod.weight = torch.nn.Parameter(quantize_per_tensor_trinary(mod.weight), requires_grad=False)
        return mod

    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: fn(mod),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )
    
class TestTrinary(QuantizationTestCase):
    