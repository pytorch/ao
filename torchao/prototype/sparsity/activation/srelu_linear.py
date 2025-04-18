
from sys import activate_stack_trampoline
import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER, 
    register_quantize_module_handler,
)
from torchao.utils import (
    is_sm_at_least_90
)
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
    _float8_cutlass_quant_sparse,
)
from torchao.core.config import AOBaseConfig


from torchao.ops import rowwise_scaled_linear_sparse_cutlass_f8f8

class ActivationLinear(nn.Linear):

    def __init__(self, *args, activation_fn=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation_fn = activation_fn

    def forward(self, x):
        if self.activation_fn:
            x = self.activation_fn(x)

        return super().forward(x)


class FFNSRelu(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w1= nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2= ActivationLinear(self.intermediate_size, self.hidden_size, bias=False, activation_fn = lambda x: F.relu(x) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.w1(x)
        y2 = self.w2(y1)
        return y2

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / factor
        out_init_std = out_init_std / factor
        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class FP8SemiSparseActivationLinear(torch.nn.Module):
    """
    Replacement nn.Linear that supports runtime fp8 activation sparsity
    """
    def __init__(self, weight) -> None:
        super().__init__()
        W_quant_func = _float8_cutlass_quant
        W_aqt = W_quant_func(weight, dtypeq_W)
        # breakpoint()
        self.Wq = W_aqt.tensor_impl.float8_data
        self.W_scale= W_aqt.tensor_impl.scale

    def forward(self, x):
        X_quant_func = _float8_cutlass_quant 
        X_aqt = X_quant_func(x, dtypeq_X)

        Xq_sparse, X_meta = None, None
        #sparse_semi_structured_tile(X_aqt.tensor_impl.float8_data, "", True)
        X_scale = X_aqt.tensor_impl.scale

        # breakpoint()

        return rowwise_scaled_linear_sparse_cutlass_f8f8(self.Wq, self.W_scale, Xq_sparse, X_meta, X_scale, bias=None, out_dtype=dtype)

    @classmethod
    def from_dense(cls, linear):
        mod = cls(linear.weight.data)
        return mod

@dataclass
class SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig(AOBaseConfig):
    """
    Applies float8 dynamic quantization to activations and float8 quantization followed by compression to sparse semi-structured tensor to weights of linear layers.

    Args:
        `layout`: layout type for quantized weight tensor, only supports `CutlassSemiSparseLayout` at the moment.
        `activation_dtype`: data type for quantized activation tensor.
        `weight_dtype`: data type for quantized weight tensor.
    """
    # layout: Layout = CutlassSemiSparseLayout()
    activation_dtype: torch.dtype = torch.float8_e5m2
    weight_dtype: torch.dtype = torch.float8_e4m3fn

@register_quantize_module_handler(SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig)
def _float8_dynamic_activation_float8_semi_sparse_weight_transform(
    module: torch.nn.Module, config: SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig
):
    assert is_sm_at_least_90(), "Float8 quantization is only supported on CUDA>=9.0"

    return module    


test = FFNSRelu(hidden_size=8192, intermediate_size=8192)
print(list(test.modules()))
