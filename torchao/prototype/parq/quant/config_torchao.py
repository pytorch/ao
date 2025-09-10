from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.dtypes import Int4CPULayout, Layout, QDQLayout
from torchao.quantization import MappingType, PerAxis, PerGroup
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.quant_api import (
    Granularity,
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    _int8_asymm_per_token_quant,
)
from torchao.quantization.quantize_.workflows import (
    IntxPackingFormat,
    IntxUnpackedToInt8Tensor,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import check_cpu_version

from .quant_api import (
    choose_qparams_stretched_affine,
    quantize_stretched_affine,
    to_stretched_affine_quantized_intx,
)
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
class Int8DynamicActivationStretchedIntxWeightConfig(AOBaseConfig):
    granularity: Granularity = PerAxis(0)
    scale_dtype: Optional[torch.dtype] = None
    layout: Layout = QDQLayout()
    version: int = 1
    b: Optional[int] = None
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    activation_quantization: Optional[str] = "int8_asym_per_token"


@register_quantize_module_handler(Int8DynamicActivationStretchedIntxWeightConfig)
def _int8_dynamic_activation_stretched_intx_transform(
    module: nn.Module, config: Int8DynamicActivationStretchedIntxWeightConfig
) -> nn.Module:
    weight = module.weight
    granularity = config.granularity
    mapping_type = MappingType.ASYMMETRIC

    assert weight.dim() == 2, (
        f"Int8DynamicActivationStretchedIntxWeightConfig only works for 2-d Tensor, got: {weight.dim()}"
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
    if config.version == 2:
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
    else:
        weight = to_stretched_affine_quantized_intx(
            *q_args,
            quant_min=config.quant_min,
            quant_max=config.quant_max,
            scale_dtype=config.scale_dtype,
            _layout=config.layout,
        )
        if config.activation_quantization == "int8_asym_per_token":
            weight = to_linear_activation_quantized(weight, _int8_asymm_per_token_quant)
        elif config.activation_quantization is not None:
            raise ValueError(f"Unsupported {config.activation_quantization=}")
    module.weight = nn.Parameter(weight, requires_grad=False)
    return module


def get_config_from_quantizer(
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
        kwargs = {"layout": Int4CPULayout()} if check_cpu_version(device) else {}
        config = Int4WeightOnlyConfig(group_size=block_size, **kwargs)
    elif isinstance(quantizer, StretchedUnifTorchaoQuantizer):
        kwargs = {"activation_quantization": None} if weight_only else {}
        config = Int8DynamicActivationStretchedIntxWeightConfig(
            b=b,
            quant_min=quantizer.quant_min,
            quant_max=quantizer.quant_max,
            granularity=granularity,
            version=version,
            **kwargs,
        )
    elif weight_only:
        config = IntxWeightOnlyConfig(
            weight_dtype=weight_dtype,
            granularity=granularity,
            mapping_type=quantizer.mapping_type,
            intx_packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
            version=version,
        )
    else:
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=weight_dtype,
            weight_granularity=granularity,
            weight_mapping_type=quantizer.mapping_type,
            act_mapping_type=MappingType.ASYMMETRIC,
            intx_packing_format=IntxPackingFormat.UNPACKED_TO_INT8,
            version=version,
        )
    return config


def is_hf_model(model: nn.Module) -> bool:
    return TRANSFORMERS_AVAIL and isinstance(
        getattr(model, "config", None), PretrainedConfig
    )


def save_hf_quantization_config(
    model: nn.Module,
    filter_fns: list[Callable[nn.Module, bool]],
    configs: list[AOBaseConfig],
) -> None:
    """Save torchao quantization config to Hugging Face model."""
    assert is_hf_model(model), "Only Hugging Face models are supported"
    assert len(filter_fns) == len(configs), (
        "filter_fns and configs must have the same length"
    )

    module_to_config = {}
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        for i, filter_fn in enumerate(filter_fns):
            if filter_fn(module):
                module_to_config[name] = configs[i]

    model.config.quantization_config = TorchAoConfig(
        quant_type=ModuleFqnToConfig(module_to_config),
        include_input_output_embeddings=True,
        modules_to_not_convert=[],
    )
