import copy
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from torchao.prototype.quantization.module_swap.data_getters import (
    DataGetter,
    get_module_input_data,
)
from torchao.prototype.quantization.module_swap.quantized_modules import (
    QuantizedLinear,
)
from torchao.prototype.quantization.module_swap.utils import (
    all_activation_quantizers_off,
    all_quantizers_off,
    all_weight_quantizers_on,
)

logger: logging.Logger = logging.getLogger(__name__)


def set_weight_min_max(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.set_weight_scale_to_min_max()


def set_weight_mse(
    model: nn.Module, num_points: int = 100, max_shrink: float = 0, norm: float = 2.0
) -> None:
    for _, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            loss_fn = lambda m: torch.sum(  # noqa
                torch.pow(torch.abs(m.weight - m.quantized_weight), norm), dim=-1
            )
            best_scale = find_optimal_scales_with_loss(
                module, loss_fn, num_points, max_shrink
            )
            module.weight_scale.data = best_scale


def get_batched_output(
    module: nn.Module, input_data: torch.Tensor, batch_size: int
) -> torch.Tensor:
    device = module.weight.device
    dtype = module.weight.dtype

    num_samples = input_data.shape[0]
    num_batches = num_samples // batch_size

    output_data = []
    for i in range(num_batches):
        this_batch = input_data[i * batch_size : (i + 1) * batch_size]
        this_batch = this_batch.to(device).to(dtype)
        output_data.append(module(this_batch).to(torch.float32).to("cpu"))

    return torch.vstack(output_data)


def set_weight_range_activation_loss(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    num_points: int = 100,
    progressive: bool = True,
    data_getter: Optional[DataGetter] = None,
) -> None:
    # store quantization settings so this algorithm does not change those implicitly
    quantization_setting_mapping_dict = {
        name: [module.weight_quantization, module.activation_quantization]
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLinear)
    }

    data_getter_progressive = None
    if data_getter is not None:
        data_getter.initialize(model, data, batch_size)
        if progressive:
            data_getter_progressive = copy.deepcopy(data_getter)

    # TODO: This can all be optimized for efficiency (keep data on GPU) or Memory (keep data on CPU)
    # Do the actual range setting
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                logger.info(f"Range setting for {name}")
                model.apply(all_quantizers_off)
                # TODO: Some form of smart subsampling from all this sequential data
                if data_getter is not None:
                    input_data = data_getter.pop(model, name)
                else:
                    input_data = get_module_input_data(model, data, module, batch_size)
                output_data = get_batched_output(module, input_data, batch_size)

                if progressive:
                    model.apply(all_weight_quantizers_on)
                    if data_getter_progressive is not None:
                        input_data = data_getter_progressive.pop(model, name)
                    else:
                        input_data = get_module_input_data(
                            model, data, module, batch_size
                        )

                input_data = input_data.to(module.weight.device).to(module.weight.dtype)
                output_data = output_data.to(module.weight.device).to(
                    module.weight.dtype
                )

                module.weight_quantization = True
                dim = tuple(range(input_data.dim() - 1))  # all but last

                # TODO: batched loss getting
                loss_fn = lambda m: torch.mean(  # noqa
                    torch.pow(m(input_data) - output_data, 2),
                    dim=dim,  # noqa
                )

                best_scale = find_optimal_scales_with_loss(module, loss_fn, num_points)
                module.weight_scale.data = best_scale

    # reset quantization settings to original values
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.weight_quantization, module.activation_quantization = (
                quantization_setting_mapping_dict[name]
            )


def set_activation_min_max(
    model: nn.Module, data: torch.Tensor, batch_size: int
) -> None:
    # store quantization settings so this algorithm does not change those implicitly
    quantization_setting_mapping_dict = {
        name: [module.weight_quantization, module.activation_quantization]
        for name, module in model.named_modules()
        if isinstance(module, QuantizedLinear)
    }

    model.apply(all_activation_quantizers_off)
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            logger.info(f"Activation min/max setting for {name}")
            input_data = None
            if module.input_quantization:
                input_data = get_module_input_data(model, data, module, batch_size)
                assert module.input_quantizer is not None
                module.input_quantizer.set_scale_offset_to_min_max(input_data)
            if module.output_quantization:
                if input_data is None:
                    input_data = get_module_input_data(model, data, module, batch_size)
                output_data = module(input_data)
                assert module.output_quantizer is not None
                module.output_quantizer.set_scale_offset_to_min_max(output_data)

    # reset quantization settings to original values
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.weight_quantization, module.activation_quantization = (
                quantization_setting_mapping_dict[name]
            )


def find_optimal_scales_with_loss(
    module: QuantizedLinear,
    loss_fn: Callable[[nn.Module], torch.Tensor],
    num_points: int,
    max_shrink: float = 0,
) -> torch.Tensor:
    assert max_shrink >= 0 and max_shrink < 1.0
    assert num_points > 0

    with torch.no_grad():
        grid = torch.linspace(max_shrink, 1, num_points + 1)
        module.set_weight_scale_to_min_max()

        orig_scales = module.weight_scale.clone()
        best_scale = module.weight_scale.clone()
        best_loss = loss_fn(module)

        for i in range(0, num_points - 1):
            test_scale = orig_scales * grid[i]
            module.weight_scale.data = test_scale
            loss = loss_fn(module)
            mask = loss < best_loss
            best_loss[mask] = loss[mask]
            best_scale[mask] = test_scale[mask]
        return best_scale


def quantize_per_group_scales(model: nn.Module, bit_width: int) -> None:
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            scale = module.weight_quantizer.scale
            assert isinstance(scale, torch.Tensor)

            if len(scale.shape) < 2 or scale.shape[-1] == 1:
                logger.warning(
                    f"Module {name} is not quantized with group_wise quantization"
                )
                continue

            per_channel_scales = torch.max(scale, dim=-1, keepdim=True).values
            scale = scale / per_channel_scales  # scale to [0, 1]

            # quantize the per_group scale to bit_width
            quant_max = 2.0**bit_width
            scale = (
                torch.clamp(torch.ceil(scale * quant_max), min=1.0, max=quant_max)
                / quant_max
            )  # ceil to make sure clipping error is certainly 0

            scale = scale * per_channel_scales  # fuse the fp16 scale

            module.weight_quantizer.scale.data = scale
