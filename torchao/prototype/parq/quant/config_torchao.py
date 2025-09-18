import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.dtypes import Int4CPULayout, Layout, QDQLayout
from torchao.quantization import MappingType, PerAxis, PerGroup
from torchao.quantization.quant_api import (
    Granularity,
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    _linear_extra_repr,
)
from torchao.quantization.quantize_.workflows import IntxUnpackedToInt8Tensor
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import check_cpu_version

from .quant_api import choose_qparams_stretched_affine, quantize_stretched_affine
from .uniform_torchao import (
    _BIT_WIDTH_TO_DTYPE,
    Int4UnifTorchaoQuantizer,
    StretchedUnifTorchaoQuantizer,
)

try:
    from transformers import PretrainedConfig, TorchAoConfig

    TRANSFORMERS_AVAIL = True
except ImportError:
    TRANSFORMERS_AVAIL = False


@dataclass
class StretchedIntxWeightConfig(AOBaseConfig):
    granularity: Granularity = PerAxis(0)
    scale_dtype: Optional[torch.dtype] = None
    layout: Layout = QDQLayout()
    version: int = 2
    b: Optional[int] = None
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    activation_quantization: Optional[str] = "int8_asym_per_token"


@register_quantize_module_handler(StretchedIntxWeightConfig)
def _int8_dynamic_activation_stretched_intx_transform(
    module: nn.Module, config: StretchedIntxWeightConfig
) -> nn.Module:
    weight = module.weight
    granularity = config.granularity
    mapping_type = MappingType.ASYMMETRIC

    if config.version != 2:
        raise NotImplementedError(f"Unsupported {config.version=}")

    assert weight.dim() == 2, (
        f"StretchedIntxWeightConfig only works for 2-d Tensor, got: {weight.dim()}"
    )
    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerAxis):
        assert granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {granularity.axis}"
        )
        group_size = weight.shape[-1]
    else:
        raise ValueError(f"granularity must be PerGroup or PerAxis, got {granularity}")

    block_size = (1, group_size)
    target_dtype = torch.int8
    q_args = (weight, mapping_type, block_size, target_dtype, config.b)
    scale, zero_point = choose_qparams_stretched_affine(
        *q_args,
        quant_min=config.quant_min,
        quant_max=config.quant_max,
    )
    qdata = quantize_stretched_affine(
        weight,
        block_size,
        scale,
        zero_point,
        target_dtype,
        quant_min=config.quant_min,
        quant_max=config.quant_max,
    )
    n_blocks = [qdata.shape[i] // block_size[i] for i in range(len(block_size))]
    scale = scale.reshape(*n_blocks)
    zero_point = zero_point.reshape(*n_blocks)

    weight = IntxUnpackedToInt8Tensor(
        qdata=qdata,
        scale=scale,
        zero_point=zero_point,
        target_dtype=getattr(torch, f"int{config.b}"),
        block_size=block_size,
        dtype=weight.dtype,
        activation_quantization=config.activation_quantization,
    )
    module.weight = nn.Parameter(weight, requires_grad=False)

    if isinstance(module, nn.Linear):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)

    return module


def _get_config_from_quantizer(
    quantizer,
    weight_only: bool,
    device: torch.device,
    b: int,
    block_size: Optional[int],
    version: int = 2,
) -> AOBaseConfig:
    granularity = PerGroup(block_size) if block_size is not None else PerAxis(0)
    weight_dtype = _BIT_WIDTH_TO_DTYPE[b]
    if isinstance(quantizer, Int4UnifTorchaoQuantizer):
        config = Int4WeightOnlyConfig(
            group_size=block_size,
            version=version,
        )
        if check_cpu_version(device):
            config.layout = Int4CPULayout()
            config.version = 1
    elif isinstance(quantizer, StretchedUnifTorchaoQuantizer):
        config = StretchedIntxWeightConfig(
            b=b,
            quant_min=quantizer.quant_min,
            quant_max=quantizer.quant_max,
            granularity=granularity,
            version=version,
        )
        if weight_only:
            config.activation_quantization = None
    elif weight_only:
        config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=quantizer.mapping_type,
            version=version,
        )
    else:
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype,
            weight_granularity=granularity,
            weight_mapping_type=quantizer.mapping_type,
            act_mapping_type=MappingType.ASYMMETRIC,
            version=version,
        )
    return config


def _is_hf_model(model: nn.Module) -> bool:
    return TRANSFORMERS_AVAIL and isinstance(
        getattr(model, "config", None), PretrainedConfig
    )


def _attach_hf_quantization_config(
    model: nn.Module,
    filter_fns: list[Callable[nn.Module, bool]],
    configs: list[AOBaseConfig],
    module_to_config: Optional[dict[str, AOBaseConfig]] = None,
) -> None:
    """Attaches torchao quantization config(s) to Hugging Face model.

    Args:
        model: nn.Module - Hugging Face model.
        filter_fns: list[Callable[nn.Module, bool]] - Callables that correspond
            to `configs`. Each `filter_fns[i]` returns whether the input module
            should be quantized with `configs[i]`. A module can map to at most
            one config.
        configs: list[AOBaseConfig] - torchao quantization configs inferred by
            `QuantOptimizer`. Each config corresponds to a param group returned
            by `optimizer.regularized_param_groups()`.
    """
    assert _is_hf_model(model), "model is not a Hugging Face model"
    assert len(filter_fns) == len(configs), (
        "filter_fns and configs must have the same length"
    )

    if module_to_config is None:
        module_to_config = {}

    seen_data_ptrs = set()
    modules_to_not_convert = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        data_ptr = module.weight.data_ptr()
        if data_ptr in seen_data_ptrs:  # do not re-quantize tied weight
            modules_to_not_convert.append(name)
            continue
        seen_data_ptrs.add(data_ptr)

        for i, filter_fn in enumerate(filter_fns):
            if filter_fn(module):
                module_to_config[name] = configs[i]

    model.config.quantization_config = TorchAoConfig(
        quant_type=ModuleFqnToConfig(module_to_config),
        include_input_output_embeddings=True,
        modules_to_not_convert=modules_to_not_convert,
    )
