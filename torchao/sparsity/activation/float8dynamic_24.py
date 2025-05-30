import types
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.dtypes import (
    CutlassSemiSparseLayout,
    Float8Layout,
    to_affine_quantized_floatx,
)
from torchao.float8.config import e4m3_dtype
from torchao.float8.inference import (
    Float8MMConfig,
    FP8Granularity,
    _check_hardware_support,
    _normalize_granularity,
)
from torchao.quantization.observer import get_block_size
from torchao.quantization.quant_api import (
    Float8Layout,
    PerRow,
    _check_hardware_support,
    _fp8_mm_compat,
    _linear_extra_repr,
    to_affine_quantized_floatx,
    to_linear_activation_quantized,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import (
    is_MI300,
    is_sm_at_least_89,
)


@dataclass
class Float8DynamicSemiSparseActivationFloat8WeightConfig(AOBaseConfig):
    """
    Configuration for applying float8 dynamic symmetric quantization + 2:4 sparsity to the activations and float8 dynamic quantization to the weights

    Args:
        activation_dtype (torch.dtype): The target data type for activation quantization. Default is torch.float8_e4m3fn.
        weight_dtype (torch.dtype): The target data type for weight quantization. Default is torch.float8_e4m3fn.
        granularity:
        The granularity for quantization. Can be either a single granularity (applied to both
            activations and weights) or a tuple of two granularities (one for activations, one for weights).
            If None, defaults to PerRowfor both. Currently both quantizations need to be the same type. And
            only PerRow is currently supported.
        mm_config (Float8MMConfig): Configuration for the matrix multiplication. Default uses fast accumulation.
        set_inductor_config (bool): if True, adjusts `torchinductor` settings to recommended values.

    """

    activation_dtype: torch.dtype = e4m3_dtype
    weight_dtype: torch.dtype = e4m3_dtype
    granularity: Optional[Union[FP8Granularity, List[FP8Granularity]]] = None
    mm_config: Optional[Float8MMConfig] = None
    set_inductor_config: bool = True

    def __post_init__(self):
        if self.mm_config is None:
            self.mm_config = Float8MMConfig(use_fast_accum=True)

        activation_granularity, weight_granularity = _normalize_granularity(
            self.granularity
        )
        self.granularity = [activation_granularity, weight_granularity]


def _float8_dynamic_sparse_activation_float8_weight_quantize_tensor(weight, config):
    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    granularity = config.granularity
    mm_config = config.mm_config

    # Ensure works on device
    _check_hardware_support(granularity)
    activation_granularity, weight_granularity = granularity

    if not _fp8_mm_compat(weight):
        return weight

    if isinstance(weight_granularity, PerRow):
        assert weight.dtype == torch.bfloat16, (
            "PerRow quantization only works for bfloat16 precision input weight"
        )

    block_size = get_block_size(weight.shape[-2:], weight_granularity)
    if weight.dim() == 3:
        block_size = tuple([1] + list(block_size))

    quantized_weight = to_affine_quantized_floatx(
        input_float=weight,
        block_size=block_size,
        target_dtype=weight_dtype,
        scale_dtype=torch.float32,
        _layout=Float8Layout(mm_config=mm_config),
    )

    # use sparsify function here instead of default fp8 quant func
    input_quant_func = _input_activation_quant_func_fp8_sparse
    input_quant_kwargs = {
        "activation_granularity": activation_granularity,
        "activation_dtype": activation_dtype,
    }

    quantized_weight = to_linear_activation_quantized(
        quantized_weight, input_quant_func, quant_kwargs=input_quant_kwargs
    )
    return quantized_weight


@register_quantize_module_handler(Float8DynamicSemiSparseActivationFloat8WeightConfig)
def _float8_dynamic_activation_sparse_float8_weight_transform(
    module: torch.nn.Module, config: Float8DynamicSemiSparseActivationFloat8WeightConfig
):
    assert is_sm_at_least_89() or is_MI300(), (
        "Float8 dynamic activation quantization is only supported on CUDA>=8.9 and MI300+"
    )
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying float8 dynamic activation quant requires module to have weight attribute"
        + f"but {module} does not have one"
    )
    quantized_weight = _float8_dynamic_sparse_activation_float8_weight_quantize_tensor(
        module.weight, config
    )
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def _input_activation_quant_func_fp8_sparse(
    x: torch.Tensor,
    activation_granularity,
    activation_dtype: torch.dtype,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
):
    """This function is used to quantize + sparsify the input activation tensor for an aqt_float variant. If scale
    is not provided it will be dynamically calculate the scales otherwise it will use the provided scale.
    """
    assert zero_point is None, (
        "Zero point is not supported for dynamic FP8 quantization"
    )

    assert isinstance(activation_granularity, PerRow), (
        "Only PerRow quantization is currently supported"
    )
    assert x.dtype == torch.bfloat16, (
        "PerRow quantization only works for bfloat16 precision input activation"
    )

    block_size = get_block_size(x.shape, activation_granularity)
    activation = to_affine_quantized_floatx(
        input_float=x,
        block_size=block_size,
        target_dtype=activation_dtype,
        scale_dtype=torch.float32,
        # we change the sparsification routine via Layout
        _layout=CutlassSemiSparseLayout(),
    )
    return activation
