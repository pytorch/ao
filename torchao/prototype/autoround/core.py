import dataclasses
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.dtypes import TensorCoreTiledLayout, to_affine_quantized_intx_static
from torchao.prototype.autoround.multi_tensor import MultiTensor, _multi_tensor_config
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.utils import find_multiple


@ar_utils.singleton
@dataclasses.dataclass
class _AutoRoundConfig:
    bits: int = 4
    group_size: int = 128
    iters: int = 200
    use_optimized_layer_output: bool = False
    gradient_accumulate_steps: int = 1
    compile_optimization_process: bool = False


_auto_round_config = _AutoRoundConfig()


@ar_utils.singleton
@dataclasses.dataclass
class _OptimizationTracker:
    num_layers: int = 0
    optimized_layers: int = 0

    def reset(self):
        self.num_layers = 0
        self.optimized_layers = 0


_optimization_tracker = _OptimizationTracker()


def _replace_model_buffers_and_params(model, replacement_fn):
    model = replacement_fn(model)
    for name, child in model.named_children():
        new_child = _replace_model_buffers_and_params(child, replacement_fn)
        if new_child is not child:
            setattr(model, name, new_child)
    return model


def _tensor_to_multi_tensor(model):
    for name, buf in model.named_buffers(recurse=False):
        setattr(model, name, MultiTensor([buf]))
    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, torch.nn.Parameter(MultiTensor([param]), False))
    return model


def _multi_tensor_to_tensor(model):
    for name, buf in model.named_buffers(recurse=False):
        if isinstance(buf, MultiTensor):
            assert (
                len(buf.values) == 1
            ), f"The buffer should only have one tensor, but got {buf.count}."
            model.register_buffer(name, buf.values[0])
    for name, param in model.named_parameters(recurse=False):
        if isinstance(param, MultiTensor):
            assert (
                len(param.values) == 1
            ), f"The parameter should only have one tensor, but got {param.count}."
            setattr(
                model, name, torch.nn.Parameter(param.values[0], requires_grad=False)
            )
    return model


@torch.no_grad()
def prepare_model_for_applying_auto_round_(
    model: torch.nn.Module,
    is_target_module: Callable[[torch.nn.Module, str], bool],
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    use_optimized_layer_output: bool = False,
    gradient_accumulate_steps: Optional[int] = 1,
    compile_optimization_process: Optional[bool] = False,
    device: Optional[torch.types.Device] = None,
):
    """Prepares the model for applying auto round optimization.

    Args:
        model (torch.nn.Module): The floating-point model to be quantized.
        is_target_module (Callable[[torch.nn.Module, str], bool]): A function that determines
            whether a module is a target module.
        bits (int, optional): The number of bits for quantization. Defaults to 4, options are 1 to 8.
        group_size (int, optional): The group size for quantization. Defaults to 128.
        iters (int, optional): The number of iterations for optimization. Defaults to 200.
        use_optimized_layer_output (bool, optional): Whether to use optimized layer output. Defaults to False.
        gradient_accumulate_steps (Optional[int]): Number of steps for accumulating gradients before
            performing the backward pass when optimizing each target module. Defaults to 1.
        compile_optimization_process (Optional[bool]): Whether to compile the optimization process. Defaults to False.
        device (Optional[torch.types.Device]): The device to use for accelrating optimization and calibration.
            Defaults to None.
    """
    _multi_tensor_config.device = device
    _multi_tensor_config.offload = next(model.parameters()).device.type != device
    _optimization_tracker.reset()

    _auto_round_config.bits = bits
    _auto_round_config.group_size = group_size
    _auto_round_config.iters = iters
    _auto_round_config.use_optimized_layer_output = use_optimized_layer_output
    _auto_round_config.gradient_accumulate_steps = gradient_accumulate_steps
    _auto_round_config.compile_optimization_process = compile_optimization_process

    logging.warning(f"config {_auto_round_config}")

    # Wrap the model buffers and parameters with `MultiTensor`
    model = _replace_model_buffers_and_params(model, _tensor_to_multi_tensor)

    def _revert_buffers_and_params_fn(
        module,
        input: Tuple[MultiTensor],
        output: Tuple[MultiTensor],
    ):
        module._forward_hook_handle_for_revert_buffers_and_params.remove()
        _replace_model_buffers_and_params(module, _multi_tensor_to_tensor)
        return output

    # Register forward hook for reverting the replacement of buffers and parameters
    model._forward_hook_handle_for_revert_buffers_and_params = (
        model.register_forward_hook(_revert_buffers_and_params_fn)
    )

    # Register forward hook for applying auto-round optimization
    def auto_round_optimization_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: Tuple[MultiTensor],
    ):
        apply_auto_round_optimization(
            module, args, kwargs, output, config=_auto_round_config
        )
        return output

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            auto_round_optimization_hook, with_kwargs=True
        )
        module._forward_hook_handle_for_auto_round = forward_hook_handle
        _optimization_tracker.num_layers += 1
        return module

    model.eval()
    ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, is_target_module
    )


def apply_auto_round():
    """Create the quantized model from the model optimized by auto-round.

    More details about the auto-round can be found at https://arxiv.org/abs/2309.05516.
    """

    raise AssertionError(
        "Please migrate this function to direct configuration, see https://github.com/pytorch/ao/issues/1690 for details"
    )

    def _apply_auto_round(optimized_model: torch.nn.Module):
        """
        The `optimized_model` includes `Linear` layers optimized by auto-round, which includes `qdq_weight`, `scale`, `zp`.
        """

        @torch.no_grad()
        def convert_weight_to_affine_quantized_tensor(observed_linear: torch.nn.Module):
            device = observed_linear.weight.device
            scale = observed_linear.scale.to(device)
            zero_point = observed_linear.zp.to(device)

            def to_uintx_weight(input_float):
                quant_min = 0
                quant_max = _auto_round_config.bits**2 - 1
                block_size = (1, observed_linear.group_size)
                from torchao.dtypes.uintx.uintx import (
                    _BIT_WIDTH_TO_DTYPE,
                    UintxLayout,
                )
                from torchao.quantization.quant_primitives import ZeroPointDomain

                assert (
                    _auto_round_config.bits in _BIT_WIDTH_TO_DTYPE
                ), f"Invalid bits: {_auto_round_config.bits}"
                dtype = _BIT_WIDTH_TO_DTYPE[_auto_round_config.bits]
                pack_dim = -1
                _layout = UintxLayout(dtype=dtype, pack_dim=pack_dim)
                return to_affine_quantized_intx_static(
                    input_float=input_float,
                    scale=scale.to(input_float.dtype),
                    zero_point=zero_point,
                    block_size=block_size,
                    target_dtype=torch.uint8,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    zero_point_domain=ZeroPointDomain.INT,
                    _layout=_layout,
                )

            def to_int4_tinygemm_weight(input_float):
                # TODO(Yi): check the weight shape, `group_size`, and `inner_k_tiles` to make sure the tinygemm can handle it
                inner_k_tiles = 8
                quant_min = 0
                quant_max = _auto_round_config.bits**2 - 1
                # Shift the `zero_point` to align with tiny gemm.
                # The dequantization process in tiny gemm:
                #   tiny_dequant = (tiny_quant - 8) * scale + tiny_zp
                # The dequantization porcess in auto-round
                #   dequant = (quant - zp) * scale
                # To align with tiny gemm:
                #   dequant = (quant - 8 + 8 - zp) * scale
                #           = (quant - 8) * scale + (8 - zp) * scale
                #              \__/                 \______________/
                #            tiny_quant                 tiny_zp
                mid_point = (quant_max + quant_min + 1) / 2
                shifted_zero_point = (mid_point - zero_point) * scale
                block_size = (1, observed_linear.group_size)
                orig_out_features, orig_in_features = input_float.shape
                in_features = find_multiple(orig_in_features, 1024)
                out_features = find_multiple(orig_out_features, 8)
                orig_num_groups = orig_in_features // observed_linear.group_size
                new_num_groups = in_features // observed_linear.group_size
                # pad scale/zero_point from [orig_out_features, orig_num_groups] to [out_features, new_num_groups]
                pad_scale = torch.nn.functional.pad(
                    scale,
                    (
                        0,
                        new_num_groups - orig_num_groups,
                        0,
                        out_features - orig_out_features,
                    ),
                )
                pad_shifted_zero_point = torch.nn.functional.pad(
                    shifted_zero_point,
                    (
                        0,
                        new_num_groups - orig_num_groups,
                        0,
                        out_features - orig_out_features,
                    ),
                )
                return to_affine_quantized_intx_static(
                    input_float=input_float,
                    scale=pad_scale.to(torch.bfloat16),
                    zero_point=pad_shifted_zero_point.to(torch.bfloat16),
                    block_size=block_size,
                    target_dtype=torch.int32,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    zero_point_domain=ZeroPointDomain.FLOAT,
                    _layout=TensorCoreTiledLayout(inner_k_tiles=inner_k_tiles),
                )

            # TODO(Yi): better way to select the weight quantization function
            if (
                _auto_round_config.bits == 4
                and observed_linear.weight.device.type == "cuda"
            ):
                weight_func = to_int4_tinygemm_weight
            else:
                weight_func = to_uintx_weight

            observed_linear.weight = torch.nn.Parameter(
                weight_func(observed_linear.weight), requires_grad=False
            )
            del observed_linear.scale
            del observed_linear.zp
            return observed_linear

        def _is_observed_linear(mod: torch.nn.Module, fqn: str):
            return hasattr(mod, "scale")

        qmodel = ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
            optimized_model,
            convert_weight_to_affine_quantized_tensor,
            _is_observed_linear,
        )
        return qmodel

    return _apply_auto_round


@torch.no_grad()
def _apply_auto_round_optimization(
    block, block_inputs, block_outputs, config: _AutoRoundConfig
):
    # Call the auto-round to execute the optimization process.
    # https://github.com/intel/auto-round/tree/patch-for-ao-2
    # TODO(Yi), make the branch more stable
    if ar_utils.is_auto_round_available():
        import auto_round
    else:
        raise ImportError(
            (
                "This example requires the `auto-round` library."
                "Please install it with `pip install git+https://github.com/intel/auto-round.git@patch-for-ao-2`"
            )
        )
    orig_device = next(block.parameters()).device
    block = block.to(_multi_tensor_config.device)
    _optimization_tracker.optimized_layers += 1
    logging.warning(
        "Apply auto-round optimization on layer %d / %d.",
        _optimization_tracker.optimized_layers,
        _optimization_tracker.num_layers,
    )

    # Start the training process to update the v, alpha and beta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,
        bits=config.bits,
        iters=config.iters,
        group_size=config.group_size,
        gradient_accumulate_steps=config.gradient_accumulate_steps,
        amp=True,
        model_dtype=next(block.parameters()).dtype,
    )
    if config.compile_optimization_process:
        rounder.quant_block_v2_ = torch.compile(rounder.quant_block_v2_)

    with torch.enable_grad():
        rounder.quant_block_v2_(
            block,
            inputs=block_inputs,
            outputs=block_outputs,
            device=_multi_tensor_config.device,
        )
    block.to(orig_device)


@ar_utils.dump_elapsed_time(record=True)
@torch.no_grad()
def apply_auto_round_optimization(
    module: torch.nn.Module,
    args: Tuple[MultiTensor],
    kwargs: Dict[str, Any],
    output: Any,
    config: _AutoRoundConfig,
):
    # Remove the hook to avoid recursive calls
    module._forward_hook_handle_for_auto_round.remove()
    # Revert the model to the original state for applying auto-round optimization
    module = _replace_model_buffers_and_params(module, _multi_tensor_to_tensor)

    block_inputs = MultiTensor.revert_to_tensor_pairs(args, kwargs)
    block_outputs = MultiTensor.revert_to_tensor_pairs(output)

    _apply_auto_round_optimization(module, block_inputs, block_outputs, config)
    # Get the new output of the optimized model
    if config.use_optimized_layer_output:
        # Re-replace the model buffers and parameters with `MultiTensor`
        _replace_model_buffers_and_params(module, _tensor_to_multi_tensor)
        output = module(*args, **kwargs)
    return output
