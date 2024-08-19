import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.dtypes import TensorCoreTiledLayoutType, to_affine_quantized_static
from torchao.prototype.autoround.multi_tensor import MultiTensor
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.utils import find_multiple
import logging
# TODO: remove it before merge
ar_utils.freeze_random()


@dataclass
class AutoRoundConfig:
    bits: int = 4
    sym: bool = False
    iters: int = 200
    group_size: int = 128
    train_bs: int = 4
    eval_bs: int = 4
    seed: int = 42
    amp: bool = True
    nsamples: int = 128
    seqlen: int = 2048
    quant_lm_head: bool = False


auto_round_config = AutoRoundConfig()


@torch.no_grad()
def create_qmodel_from_qdq_model(qdq_model: torch.nn.Module):
    """Create a quantized model from a qdq model.

    The qdq_model includes Linear quantized by auto-round, which includes qdq weight, scale, zp.
    """

    @torch.no_grad()
    def apply_static_quant(observed_linear):
        device = observed_linear.weight.device
        scale = observed_linear.scale.to(device)
        zero_point = observed_linear.zp.to(device)

        def to_uintx_weight(input_float):
            quant_min = 0
            quant_max = auto_round_config.bits**2 - 1
            block_size = (1, observed_linear.group_size)
            from torchao.dtypes.uintx.Uintx import UintxLayoutType
            from torchao.quantization.quant_primitives import (
                MappingType,
                ZeroPointDomain,
            )

            pack_dim = -1
            bit_width = auto_round_config.bits
            layout_type = UintxLayoutType(bit_width=bit_width, pack_dim=pack_dim)
            return to_affine_quantized_static(
                input_float=input_float,
                scale=scale.to(torch.bfloat16),
                zero_point=zero_point,
                block_size=block_size,
                target_dtype=torch.uint8,
                quant_min=quant_min,
                quant_max=quant_max,
                zero_point_domain=ZeroPointDomain.INT,
                layout_type=layout_type,
            )

        def to_int4_tinygemm_weight(input_float):
            # TODO(Yi): check the weight shape, `group_size`, and `inner_k_tiles` to make sure the tinygemm can handle it
            inner_k_tiles = 8
            quant_min = 0
            quant_max = auto_round_config.bits**2 - 1
            # Shift the zeros to align with tiny gemm.
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
            return to_affine_quantized_static(
                input_float=input_float,
                scale=pad_scale.to(torch.bfloat16),
                zero_point=pad_shifted_zero_point.to(torch.bfloat16),
                block_size=block_size,
                target_dtype=torch.int32,
                quant_min=quant_min,
                quant_max=quant_max,
                zero_point_domain=ZeroPointDomain.FLOAT,
                layout_type=TensorCoreTiledLayoutType(inner_k_tiles=inner_k_tiles),
            )

        # TODO(Yi): better way to select the weight quantization function
        if auto_round_config.bits == 4:
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
        qdq_model, apply_static_quant, _is_observed_linear
    )
    return qmodel

layer_idx = 0
@ar_utils.dump_elapsed_time()
@torch.no_grad()
def apply_auto_round(block, grouped_args, spec, block_outputs):
    # Call the auto-round to execute the optimization process
    import auto_round
    
    global layer_idx
    layer_idx += 1
    logging.info(f"Apply auto-round for layer {layer_idx}")

    ar_utils.see_memory_usage("Before apply auto-round")

    global auto_round_config

    # Start the training process to update the v, alpha and betta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=auto_round_config.sym,  # Both True and False are OK
        bits=auto_round_config.bits,
        iters=auto_round_config.iters,
        use_quant_input=False,  # disable it for now
        amp=auto_round_config.amp,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )

    @torch.no_grad()
    def _unflatten_grouped_args(grouped_args, spec):
        inputs = []
        for i, inp in enumerate(grouped_args):
            cur_args, cur_kwargs = tree_unflatten(inp, spec)
            inputs.append((cur_args, cur_kwargs))
        return inputs

    block_inputs = _unflatten_grouped_args(grouped_args, spec)
    with torch.enable_grad():
        rounder.quant_block_v2_(
            block, inputs=block_inputs, outputs=block_outputs, device="cuda"
        )
    # TODO(Yi): move block to cpu will cause the accuracy issue.
    block = create_qmodel_from_qdq_model(block)
    ar_utils.see_memory_usage("After apply auto-round.")
    return block


@torch.no_grad()
def optimize_module(
    module: torch.nn.Module,
    args: Tuple[MultiTensor],
    kwargs: Dict[str, MultiTensor],
    output: Tuple[MultiTensor],
):
    # Remove the hook before otpimization process to avoid the recursive call
    module._forward_hook_handle_for_autoround.remove()
    flat_args, spec = tree_flatten((args, kwargs))
    grouped_args = MultiTensor.flat_to_grouped(flat_args)
    output_flat_args, output_spec = tree_flatten((output, {}))
    output_grouped_args = MultiTensor.flat_to_grouped(output_flat_args)
    apply_auto_round(module, grouped_args, spec, output_grouped_args)
    torch.cuda.empty_cache()


@torch.no_grad()
def prepare_model_for_applying_auto_round_(model, is_decoder):

    def forward_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: Tuple[MultiTensor],
    ):
        optimize_module(module, args, kwargs, output)
        return output

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            forward_hook, with_kwargs=True
        )
        module._forward_hook_handle_for_autoround = forward_hook_handle
        return module

    ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, is_decoder
    )
