from dataclasses import dataclass

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
)
from torchao.quantization.quant_api import (
    _float8_cutlass_quant,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


@dataclass
class SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig(AOBaseConfig):
    """
    Applies float8 dynamic quantization to activations and float8 quantization followed by compression to sparse semi-structured tensor to weights of linear layers.

    Args:
        `activation_dtype`: data type for quantized activation tensor.
        `weight_dtype`: data type for quantized weight tensor.
    """

    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn


@register_quantize_module_handler(
    SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig
)
def _float8_dynamic_activation_float8_semi_sparse_weight_transform(
    module: torch.nn.Module,
    config: SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig,
):
    return FP8SemiSparseActivationLinear.from_dense(module, config)


class FP8SemiSparseActivationLinear(nn.Module):
    """
    Replacement nn.Linear that supports runtime fp8 activation sparsity
    """

    def __init__(self, weight, config) -> None:
        super().__init__()
        self.config = config

        W_aqt = _float8_cutlass_quant(weight, self.config.weight_dtype)
        self.Wq = W_aqt.tensor_impl.float8_data
        self.W_scale = W_aqt.tensor_impl.scale

    def forward(self, x):
        X_scale = torch.empty([x.shape[0], 1], device=x.device, dtype=torch.float32)
        Xq_sparse, X_meta = torch.ops.torchao.sparse24_sm90_sparsify(
            x,
            "cutlass",
            "srelu",
            "largest",
            dtype=self.config.activation_dtype,
            scale=X_scale,
        )

        result = rowwise_scaled_linear_sparse_cutlass_f8f8(
            self.Wq,
            self.W_scale,
            Xq_sparse,
            X_meta,
            X_scale,
            bias=None,
            out_dtype=torch.bfloat16,
        ).t()

        return result

    @classmethod
    def from_dense(
        cls, linear, config: SRELUFloat8SemiSparseDynamicActivationFloat8WeightConfig
    ):
        if linear.bias is not None:
            raise NotImplementedError("bias is not supported")
        if linear.weight.dtype != torch.bfloat16:
            raise NotImplementedError("weight dtype must be bf16")

        return cls(linear.weight.data, config)


@dataclass
class ActivationSparseLinearConfig(AOBaseConfig):
    """
    Adds in acceleration for activation sparsity to linear layers for decode. 

    Args:
        `activation_dtype`: data type for quantized activation tensor.
        `weight_dtype`: data type for quantized weight tensor.
    """

    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn

@register_quantize_module_handler(
    ActivationSparseLinearConfig
)
def _activation_spare_linear_transform(
    module: torch.nn.Module,
    config: ActivationSparseLinearConfig,
):
    return ActivationSparseLinear.from_dense(module, config)


class ActivationSparseLinear(nn.Module):
    """
    Replacement nn.Linear that supports runtime fp8 activation sparsity
    """

    def __init__(self, weight, config) -> None:
        super().__init__()
        self.config = config
        self.weight_transposed = weight.T.contiguous().T

    def forward(self, x):
        if x.shape[1] == 1:
            return torch.ops.torchao.splitk_sparse_gemv(x, self.weight_transposed)
        else:
            return torch.nn.functional.linear(x, self.weight_transposed)

    @classmethod
    def from_dense(
        cls, linear, config:ActivationSparseLinearConfig 
    ):
        if linear.bias is not None:
            raise NotImplementedError("bias is not supported")
        return cls(linear.weight.data, config)
