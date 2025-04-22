from dataclasses import dataclass
from sys import activate_stack_trampoline

import torch
import torch.nn.functional as F
from torch import nn
from torchao.core.config import AOBaseConfig

from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
)
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
)

from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
    register_quantize_module_handler,
)
from torchao.utils import is_sm_at_least_90

SUPPORTED_ACTIVATION_FUNCTIONS = {
    None: lambda x: x,
    "srelu": lambda x: (F.relu(x) ** 2), 
}

class LoggerLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # breakpoint()
        print("logging: ", x)
        return super().forward(x)

class SquaredReLUFFNDense(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w2 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, activation_fn="srelu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.w1(x)
        y2 = F.relu(y1) ** 2
        y2 = self.w2(y2)
        return y2

    def reset_parameters(self, init_std=None, factor=2.0):
        in_init_std = init_std or (self.dim ** (1.5))
        out_init_std = init_std or (self.hidden_dim ** (1.5))
        in_init_std = in_init_std / factor
        out_init_std = out_init_std / factor
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=1.0,
            std=in_init_std,
            a=-2 * in_init_std,
            b=4 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=1.0,
            std=out_init_std,
            a=-2 * out_init_std,
            b=4 * out_init_std,
        )


class FP8SemiSparseActivationLinear(nn.Module):
    """
    Replacement nn.Linear that supports runtime fp8 activation sparsity
    """

    def __init__(self, weight) -> None:
        super().__init__()
        W_aqt = _float8_cutlass_quant(weight, torch.float8_e4m3fn)
        # self.Wq = W_aqt.tensor_impl.float8_data.T
        # self.W_scale = W_aqt.tensor_impl.scale.unsqueeze(-1).T
        # breakpoint()
        self.Wq = W_aqt.tensor_impl.float8_data
        self.W_scale = W_aqt.tensor_impl.scale

    def forward(self, x):

        X_scale = torch.empty([x.shape[0], 1], device=x.device, dtype=torch.float32)
        Xq_sparse, X_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            x,
            "cutlass",
            "srelu",
            "largest",
            dtype=torch.float8_e4m3fn,
            scale=X_scale,
        )
        # print("reference scales:", X_aqt.tensor_impl.scale)
        # print("new scales:", X_scale.squeeze(-1))
        # torch.testing.assert_close(X_scale, X_aqt.tensor_impl.scale.unsqueeze(-1))

        res = rowwise_scaled_linear_sparse_cutlass_f8f8(
            self.Wq,
            self.W_scale,
            Xq_sparse,
            X_meta,
            X_scale,
            bias=None,
            out_dtype=torch.bfloat16,
        ).t()
        return res

    @classmethod
    def from_dense(cls, linear):
        return cls(linear.weight.data)


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
    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn


@register_quantize_module_handler(
    SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig
)
def _float8_dynamic_activation_float8_semi_sparse_weight_transform(
    module: torch.nn.Module,
    config: SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig,
):
    assert is_sm_at_least_90(), "Float8 quantization is only supported on CUDA>=9.0"

    return module
